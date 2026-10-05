"""Surround facilitation as an artifact of the MEI-and-mask procedure.

Four modules:

    models      Model A (LN Gabor), Model B (Heeger DN), Model C (DN plus a
                multiplicative facilitatory gain from an orthogonal annular pool).
    mask        MEI -> z-score -> threshold -> close -> largest component ->
                convex hull -> blur, plus dilation and the analytic headroom bound.
    optimize    MEI optimization and surround optimization. 
    measure     Drive, response, facilitation, annulus construction.

Sweep drivers compute and write payloads; figure scripts read payloads and never
compute. Full regeneration takes over an hour, so that separation is not negotiable.
"""

__version__ = "0.1.0"
