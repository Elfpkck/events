class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
        self.__opacity = 0.3
        self.coef = 0.2

    @staticmethod
    def get_black_color():
        return 0, 0, 0

    @classmethod
    def get_black_instance(cls):
        return cls(0, 0, 0)

    @property
    def opacity(self):
        return self.__opacity * self.coef


class Color2(Color):
    def __init__(self, r, g, b):
        super().__init__(r, g, b)
        self.r = 5


if __name__ == '__main__':
    # staticmethod
    print(Color.get_black_color())
    print(Color(12, 155, 45).get_black_color())

    # classmethod
    print(Color.get_black_instance().r)
    print(Color2.get_black_instance().r)
    print(Color(4, 5, 6).get_black_instance().r)
    print(Color2(7, 7, 7).get_black_instance().r)

    # property
    # Color(0, 0, 0).opacity = 5  # AttributeError
    print(Color(0, 0, 0).opacity)
