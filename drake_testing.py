def reexecute_if_unbuffered():
    """Ensures that output is immediately flushed (e.g. for segfaults).
    ONLY use this at your entrypoint. Otherwise, you may have code be
    re-executed that will clutter your console."""
    import os
    import shlex
    import sys
    if os.environ.get("PYTHONUNBUFFERED") in (None, ""):
        os.environ["PYTHONUNBUFFERED"] = "1"
        argv = list(sys.argv)
        if argv[0] != sys.executable:
            argv.insert(0, sys.executable)
        cmd = " ".join([shlex.quote(arg) for arg in argv])
        sys.stdout.flush()
        os.execv(argv[0], argv)


def traced(func, ignoredirs=None):
    """Decorates func such that its execution is traced, but filters out any
     Python code outside of the system prefix."""
    import functools
    import sys
    import trace
    if ignoredirs is None:
        ignoredirs = ["/usr", sys.prefix]
    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return tracer.runfunc(func, *args, **kwargs)

    return wrapped


# NOTE: You don't have to trace all of your code. If you can identify a
# single function, then you can just decorate it with this. If you're
# decorating a class method, then be sure to declare these functions above
# it.
@traced
def main():
    from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, FindResourceOrThrow,
    Parser, Simulator)

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
    Parser(plant).AddModelFromFile(
    FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
    plant.Finalize()
    diagram = builder.Build()
    simulator = Simulator(diagram)

if __name__ == "__main__":
    reexecute_if_unbuffered()
    main()