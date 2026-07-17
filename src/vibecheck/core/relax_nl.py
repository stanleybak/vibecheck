"""Sound closed-form relaxations for elementwise Sin/Cos/Pow.

Ported verbatim from v1 (vibecheck.nonlinear_relax + nl_sin/nl_cos/nl_pow)
as part of making vibecheck STANDALONE; pruned to the surface vc2 uses
(coverage-audited): `affine_band(lo, hi, lam=None)` and
`affine_band_alpha(lo, hi, alpha)` via `slope_at` -- the RelaxLib _V1Band
adapter (core/relax.py) is the only caller. The v1 interval()/curvature()
methods, the op REGISTRY, and the zono transformer were dead here and are
not copied.

SOUNDNESS IS BY CONSTRUCTION -- closed form / monotonicity / convexity /
critical-point (root of f'(x) = lam) analysis. NEVER sample to derive a
bound (the worst case can lie between samples); sampling only VALIDATES
bands in tests.
"""
import math

import torch

_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi


class ScalarNonlinearRelax:
    """Base class for an elementwise scalar-nonlinearity relaxation.

    Subclasses MUST implement `func`, `interval`, and `affine_band`. All three
    operate element-wise and broadcast; `lo`/`hi` are torch tensors and the
    returned tensors share their shape.
    """
    onnx_op = None

    def func(self, x):
        """Exact elementwise function (torch tensor -> torch tensor)."""
        raise NotImplementedError

    def affine_band(self, lo, hi, lam=None):
        """Sound affine over-approximation. Returns (lam, mu, delta) with
        |f(x) - (lam*x + mu)| <= delta  for all x in [lo, hi], delta >= 0.
        Must be sound by construction (no sampling).

        ``lam`` (optional): use THIS slope instead of the default chord slope.
        ANY real lam is sound — mu/delta are recomputed as the exact midpoint /
        half-range of g(x)=f(x)-lam*x over [lo,hi] (endpoints + critical points
        where f'(x)=lam). This is the α-CROWN hook: a differentiable lam lets
        gradient pick the slope that maximises the downstream margin."""
        raise NotImplementedError

    def slope_at(self, x):
        """f'(x) — used by ``affine_band_alpha`` to span a sound slope range."""
        raise NotImplementedError

    def affine_band_alpha(self, lo, hi, alpha):
        """α-parametrised band: lam = (1-α)·f'(lo) + α·f'(hi), α in [0,1].
        Differentiable in α; sound for every α (``affine_band`` recomputes a
        sound mu/delta for whatever lam it is given). Mirrors the convex-op
        α-zono path (exp/reciprocal) used by ``_zono_alpha_close``."""
        a = alpha.clamp(0.0, 1.0) if torch.is_tensor(alpha) else alpha
        d_lo = self.slope_at(lo)
        d_hi = self.slope_at(hi)
        lam = d_lo + a * (d_hi - d_lo)
        return self.affine_band(lo, hi, lam=lam)


def _interval_contains_congruent(lo, hi, theta):
    """Element-wise boolean: does [lo, hi] contain some x congruent to ``theta``
    modulo 2*pi, i.e. exists integer k with lo <= theta + 2*pi*k <= hi?

    Equivalent to floor((hi - theta) / 2pi) >= ceil((lo - theta) / 2pi).
    Vectorized over the broadcast shape of lo/hi.
    """
    k_hi = torch.floor((hi - theta) / _TWO_PI)
    k_lo = torch.ceil((lo - theta) / _TWO_PI)
    return k_hi >= k_lo



class SinRelax(ScalarNonlinearRelax):
    """Sound relaxation for elementwise sin."""

    def func(self, x):
        return torch.sin(x)

    def slope_at(self, x):
        return torch.cos(torch.as_tensor(x, dtype=torch.float64))

    def affine_band(self, lo, hi, lam=None):
        """Sound affine band (lam, mu, delta): |sin(x) - (lam*x + mu)| <= delta
        for all x in [lo, hi].

        lam = chord slope (sin(hi) - sin(lo)) / (hi - lo), or cos(lo) when
        hi == lo (or a caller-supplied α-CROWN slope). The deviation
        g(x) = sin(x) - lam*x is smooth, so its max and min over [lo, hi] occur
        at the endpoints or at interior stationary points where
        g'(x) = cos(x) - lam = 0, i.e. cos(x) = lam. Those are
        x = +-arccos(lam) + 2*pi*k. We enumerate every such x inside [lo, hi]
        (a bounded count: at most ~ (hi - lo)/(2*pi) + 2 per branch), evaluate g
        at lo, hi and each in-range critical point, then set
        mu = (gmax + gmin)/2, delta = (gmax - gmin)/2. Sound for ANY lam.
        """
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        lo = lo.contiguous()
        hi = hi.contiguous()

        width = hi - lo
        degenerate = width <= 0.0
        if lam is None:
            # chord slope; on hi == lo use the local derivative cos(lo).
            denom = torch.where(degenerate, torch.ones_like(width), width)
            lam = torch.where(degenerate,
                              torch.cos(lo),
                              (torch.sin(hi) - torch.sin(lo)) / denom)

        def g(x):
            return torch.sin(x) - lam * x

        # Start the running max/min from the two endpoints.
        g_lo = g(lo)
        g_hi = g(hi)
        gmax = torch.maximum(g_lo, g_hi)
        gmin = torch.minimum(g_lo, g_hi)

        # Interior stationary points: cos(x) = lam. Real only when |lam| <= 1.
        has_root = lam.abs() <= 1.0
        lam_clamped = lam.clamp(-1.0, 1.0)
        base = torch.arccos(lam_clamped)  # principal value in [0, pi]

        # The two arccos branches: x = +base + 2*pi*k and x = -base + 2*pi*k.
        # Bound the integer-k range by the interval width.
        max_periods = int(math.floor(float(width.max().item()) / _TWO_PI)) + 2 \
            if width.numel() > 0 else 2

        for sign in (1.0, -1.0):
            theta = sign * base  # element-wise candidate phase in [-pi, pi]
            # Smallest k such that theta + 2*pi*k >= lo:
            k_start = torch.ceil((lo - theta) / _TWO_PI)
            for j in range(max_periods + 1):
                k = k_start + j
                xc = theta + _TWO_PI * k
                in_range = has_root & (xc >= lo) & (xc <= hi)
                if not bool(in_range.any()):
                    continue
                gc = g(xc)
                # Only let in-range critical points move the extrema.
                gmax = torch.where(in_range, torch.maximum(gmax, gc), gmax)
                gmin = torch.where(in_range, torch.minimum(gmin, gc), gmin)

        mu = 0.5 * (gmax + gmin)
        delta = 0.5 * (gmax - gmin)
        delta = delta.clamp_min(0.0)
        return lam, mu, delta


class CosRelax(ScalarNonlinearRelax):
    """Sound relaxation for elementwise cos."""

    def func(self, x):
        return torch.cos(x)

    def slope_at(self, x):
        return -torch.sin(torch.as_tensor(x, dtype=torch.float64))

    def affine_band(self, lo, hi, lam=None):
        """Sound affine band around the chord (or a caller-supplied α-CROWN
        slope — sound for ANY lam).

        Let lam be the chord slope and g(x) = cos(x) - lam*x. g is smooth, so on
        the closed interval [lo, hi] its extrema occur at the endpoints or at
        stationary points g'(x) = -sin(x) - lam = 0  <=>  sin(x) = -lam.

        Stationary points (when |lam| <= 1):
            x = -arcsin(lam) + 2*pi*k          (from sin(x) = -lam)
            x = pi + arcsin(lam) + 2*pi*k
        We enumerate every integer k that can place such a point in [lo, hi]
        (the count of full periods is (hi - lo)/(2*pi); +2 slack covers the
        partial periods at both ends and arcsin offsets), evaluate g at each
        in-range stationary point and at both endpoints, take gmax/gmin, then
            mu    = (gmax + gmin) / 2
            delta = (gmax - gmin) / 2
        so  gmin <= cos(x) - lam*x <= gmax  =>  |cos(x) - (lam*x + mu)| <= delta
        for all x in [lo, hi]. Exact (tightest for this lam) and sound.
        """
        lo = torch.as_tensor(lo)
        hi = torch.as_tensor(hi)
        lo, hi = torch.broadcast_tensors(lo, hi)
        work_dtype = lo.dtype if lo.dtype.is_floating_point else torch.float64
        lo = lo.to(work_dtype)
        hi = hi.to(work_dtype)

        width = hi - lo
        degenerate = width.abs() <= 0.0  # hi == lo

        if lam is None:
            # Chord slope; guard hi == lo with the exact derivative -sin(lo).
            safe_width = torch.where(degenerate, torch.ones_like(width), width)
            lam_chord = (torch.cos(hi) - torch.cos(lo)) / safe_width
            lam = torch.where(degenerate, -torch.sin(lo), lam_chord)
        else:
            lam = torch.as_tensor(lam, dtype=work_dtype, device=lo.device)

        def g(x):
            return torch.cos(x) - lam * x

        g_lo = g(lo)
        g_hi = g(hi)
        gmax = torch.maximum(g_lo, g_hi)
        gmin = torch.minimum(g_lo, g_hi)

        # Stationary points need a real arcsin(lam); clamp keeps arcsin finite,
        # the `valid` mask below discards contributions where |lam| > 1.
        valid = lam.abs() <= 1.0
        asin = torch.asin(torch.clamp(lam, -1.0, 1.0))
        base_a = -asin                # x = -arcsin(lam) + 2*pi*k
        base_b = math.pi + asin       # x = pi + arcsin(lam) + 2*pi*k

        # Anchor the k-sweep to the interval's LOCATION (not just its width):
        # for each stationary-point family `base + 2*pi*k`, the integer k that
        # lands the point nearest the interval centre is
        #     k0(elt) = round((centre - base) / (2*pi))
        # which is a per-element tensor (an x ~ 30 interval needs k ~ 5, an
        # x ~ -30 interval needs k ~ -5 — a width-only cap misses both). We then
        # sweep a small WINDOW of integer offsets around k0 wide enough to span
        # the periods inside [lo, hi]. The interval covers width/(2*pi) periods,
        # so +-(ceil(width/2*pi) + 1) offsets from k0 is guaranteed to enumerate
        # every in-range stationary point of each family. Per-element [lo, hi]
        # masking discards out-of-range candidates.
        centre = (lo + hi) * 0.5
        half_win = int(math.ceil(float((width.abs() / _TWO_PI).max().item()))) + 1 \
            if width.numel() else 1
        neg_inf = torch.full_like(lo, float('-inf'))
        pos_inf = torch.full_like(lo, float('inf'))
        for base in (base_a, base_b):
            k0 = torch.round((centre - base) / _TWO_PI)
            for d in range(-half_win, half_win + 1):
                x = base + _TWO_PI * (k0 + d)
                in_range = valid & (x >= lo) & (x <= hi)
                gx = g(x)
                cand_max = torch.where(in_range, gx, neg_inf)
                cand_min = torch.where(in_range, gx, pos_inf)
                gmax = torch.maximum(gmax, cand_max)
                gmin = torch.minimum(gmin, cand_min)

        mu = (gmax + gmin) * 0.5
        delta = (gmax - gmin) * 0.5
        delta = torch.clamp(delta, min=0.0)
        return lam, mu, delta


class PowRelax(ScalarNonlinearRelax):
    """Sound relaxation for elementwise x**p, integer p >= 2."""

    def __init__(self, p=2):
        p_int = int(p)
        if p_int != p:
            raise ValueError(f'PowRelax requires an integer exponent, got {p!r}')
        if p_int < 2:
            raise ValueError(f'PowRelax requires p >= 2, got {p_int}')
        self.p = p_int

    def func(self, x):
        return x ** self.p

    def slope_at(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        return self.p * x ** (self.p - 1)

    def affine_band(self, lo, hi, lam=None):
        """Sound affine band (lam, mu, delta): |x**p - (lam*x + mu)| <= delta
        for all x in [lo, hi]. Sound for ANY lam (α-CROWN slope override).

        lam defaults to the chord slope (hi**p - lo**p)/(hi - lo); on the
        degenerate hi == lo the band collapses and lam = p*lo**(p-1).
        g(x) = x**p - lam*x is smooth; its extrema over [lo, hi] are at the
        endpoints or at stationary points x**(p-1) = lam/p. We enumerate those
        roots in-range and bracket gmin/gmax over {lo, hi, roots}.
        """
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        lo = lo.contiguous()
        hi = hi.contiguous()

        p = self.p
        width = hi - lo
        degenerate = width <= 0.0
        if lam is None:
            denom = torch.where(degenerate, torch.ones_like(width), width)
            lam = torch.where(degenerate,
                              p * lo ** (p - 1),
                              (hi ** p - lo ** p) / denom)

        def g(x):
            return x ** p - lam * x

        # Endpoints seed the running bracket.
        g_lo = g(lo)
        g_hi = g(hi)
        gmax = torch.maximum(g_lo, g_hi)
        gmin = torch.minimum(g_lo, g_hi)

        # Stationary points: x**(p-1) = lam/p.  m = |lam/p|**(1/(p-1)) >= 0.
        # Compute the magnitude on the non-negative |lam/p| (avoids a fractional
        # power of a negative base in torch) and reattach the sign per branch.
        ratio = lam / p
        m = ratio.abs().pow(1.0 / (p - 1))

        if p % 2 == 1:
            # p-1 even: x**(p-1) >= 0, so a real root needs lam/p >= 0, and then
            # BOTH x = +m and x = -m solve it.
            real = ratio >= 0.0
            candidates = (m, -m)
        else:
            # p-1 odd: x**(p-1) is odd/monotone -> exactly one real root,
            # x = sign(lam/p) * m.
            real = torch.ones_like(lam, dtype=torch.bool)
            xc_even = torch.sign(ratio) * m
            candidates = (xc_even,)

        for xc in candidates:
            in_range = real & (xc >= lo) & (xc <= hi)
            if not bool(in_range.any()):
                continue
            gc = g(xc)
            gmax = torch.where(in_range, torch.maximum(gmax, gc), gmax)
            gmin = torch.where(in_range, torch.minimum(gmin, gc), gmin)

        mu = 0.5 * (gmax + gmin)
        delta = 0.5 * (gmax - gmin)
        delta = delta.clamp_min(0.0)
        return lam, mu, delta
