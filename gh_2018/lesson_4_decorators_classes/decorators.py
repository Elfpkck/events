def additional_count(func):
    def wrapper(x):
        return func(x) * 2
    return wrapper


@additional_count
def count(x: int) -> int:
    return x + x


def print_class_method_names(method):
    """Prints name of class and name of current method at the beginning of the call."""
    def wrapper(*args, **kwargs):
        print(f"{args[0].__class__.__name__} {method.__name__} is called")
        return method(*args, **kwargs)
    return wrapper


if __name__ == '__main__':
    class AA:
        @print_class_method_names
        def abc(self):
            pass

        @print_class_method_names
        def bbbb(self):
            pass

    print(count(2))
    AA().abc()
    AA().bbbb()
