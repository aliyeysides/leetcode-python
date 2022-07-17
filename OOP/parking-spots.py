from enum import Enum, auto
from typing import List

"""
INPUT::::
n = 5
instructions = [
  ["park", "1", "Small", "Silver", "BMW"],
  ["park", "1", "Large", "Black", "Nissan"],
  ["print", "1"],
  ["print", "2"],
  ["print", "3"],
]

OUTPUT::::
[
  "Small Silver BMW",
  "Large Black Nissan",
  "Empty",
]
"""


class Size(Enum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()


class Car:
    SIZES = {
        'Small': Size.SMALL,
        'Medium': Size.MEDIUM,
        'Large': Size.LARGE
    }

    SIZE_LABELS = {e: n for n, e in SIZES.items()}

    def __init__(self, size: str, color: str, brand: str):
        self.__size = self.SIZES[size]
        self.__color = color
        self.__brand = brand

    def __str__(self) -> str:
        return f"{self.SIZE_LABELS[self.__size]} {self.__color} {self.__brand}"


class ParkingLot:
    def __init__(self, spots: int):
        self.__spots = [0] * spots

    def park(self, spot: int, car: Car):
        if self.__spots[spot] == 0:
            self.__spots[spot] = car
        # else:
            # for space in self.__spots

    def __str__(self) -> str:
        return f"{self.__spots}"


def parking_system(n: int, instructions: List[List[str]]) -> List[str]:
    system = ParkingLot()
    for instruction in instructions:
        command = instruction[0]
        if command == 'park':
            spot = instruction[1]
            car_size = instruction[2]
            car_color = instruction[3]
            car_brand = instruction[4]
            car = Car(car_size, car_color, car_brand)
            system.park(spot, car)


if __name__ == '__main__':
    n = int(input())
    instructions = [input().split() for _ in range(int(input()))]
    res = parking_system(n, instructions)
    for line in res:
        print(line)
