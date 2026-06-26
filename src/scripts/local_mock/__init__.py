# Local-only mock package for the dev sandbox env wrapper.
# Intentionally named to never collide with any real Isaac Lab / Omniverse package,
# since Python prepends the script's own directory to sys.path and a colliding
# name here would silently shadow the real package on an AWS Isaac Sim instance.
