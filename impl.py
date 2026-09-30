"""
impl.py - one tiny switch that chooses WHICH version of a function runs.

Why this file exists
    In v2 I added an "alternative function" next to every important step, for example:
        cv2.threshold(...)          -> skimage.filters.threshold_otsu(...)
        hand written RANSAC         -> cv2.estimateAffine2D(..., method=cv2.RANSAC)
        np.linalg.svd(...)          -> sklearn.decomposition.PCA(...)
    Both versions do the SAME job, they are just written by different libraries.
    The switch below lets you flip between all of them with ONE word, so you can
    compare them on the same video without editing any code.

        MODE = "base"  -> the original functions are used (default, this is what runs)
        MODE = "alt"   -> the alternative functions are used

    How to change it
        1) From the command line, every script accepts the same flag:
               python pipeline.py --impl alt
               python moduleC/optical_flow.py --impl alt
        2) By hand, in any python shell:
               import impl; impl.set_mode("alt")
        3) Or permanently, by editing MODE below.

Important
    "base" mode gives exactly the same numbers as the original project, so it is the
    mode you use when you want to compare old and new results. "alt" mode gives very
    similar numbers, but not always bit-for-bit identical, because two different
    libraries round their maths in a slightly different way.
"""

# The current mode. Every module reads this value through is_alt().
MODE = "base"

# The two allowed values, kept here so every script can build the same --impl menu.
CHOICES = ("base", "alt")


def set_mode(mode):
    """Change the mode for the whole project. Returns the mode that was set."""
    global MODE
    mode = str(mode).lower()
    if mode not in CHOICES:
        raise ValueError(f"unknown impl mode {mode!r}, pick one of {CHOICES}")
    MODE = mode
    return MODE


def get_mode():
    """Read the current mode without changing it."""
    return MODE


def is_alt():
    """True when the alternative functions should run, False for the original ones."""
    return get_mode() == "alt"


def add_impl_flag(parser):
    """Add the shared --impl flag to an argparse parser. Returns the same parser."""
    parser.add_argument("--impl", choices=CHOICES, default=MODE,
                        help="base = original functions, alt = alternative functions")
    return parser


def apply_args(args):
    """Call this right after parse_args() so the flag really takes effect."""
    return set_mode(getattr(args, "impl", MODE))
