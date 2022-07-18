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


SIZES = {
    'Small': Size.SMALL,
    'Medium': Size.MEDIUM,
    'Large': Size.LARGE
}

SIZE_LABELS = {e: n for n, e in SIZES.items()}


class Car:
    def __init__(self, size: str, color: str, brand: str):
        self.__size = SIZES[size]
        self.__color = color
        self.__brand = brand

    def __str__(self) -> str:
        return f"{SIZE_LABELS[self.__size]} {self.__color} {self.__brand}"


class ParkingSpot:
    def __init__(self):
        self.parked_car = None

    def park(self, car: Car) -> bool:
        if self.parked_car:
            return False
        self.parked_car = car
        return True

    def leave(self) -> bool:
        if self.parked_car:
            self.parked_car = None
            return True
        return False

    def __str__(self) -> str:
        if self.parked_car:
            return str(self.parked_car)
        return "Empty"


class ParkingLot:
    def __init__(self, size: int):
        self.spots: list[ParkingSpot] = []
        self.__size = size
        self.__free_spots = size
        for _ in range(size):
            self.spots.append(ParkingSpot())

    def park(self, spot: int, car: Car) -> bool:
        for space in range(spot, self.__size):
            if self.spots[space].park(car):
                self.__free_spots -= 1
                return True
        return False

    def leave(self, spot: int) -> None:
        if self.spots[spot].leave():
            self.__free_spots += 1

    def free_spots(self) -> int:
        return self.__free_spots

    def __str__(self) -> str:
        return f"{self.spots}"


def parking_system(n: int, instructions: List[List[str]]) -> List[str]:
    parking_lot = ParkingLot(n)
    ans = []
    for instruction in instructions:
        command, *args = instruction
        if command == 'park':
            spot, *car = args
            parking_lot.park(int(spot), Car(*car))
        elif command == "remove":
            parking_lot.leave(int(args[0]))
        elif command == "print":
            ans.append(str(parking_lot.spots[int(args[0])]))
        elif command == "print_free_spots":
            ans.append(str(parking_lot.free_spots()))
    return ans


if __name__ == '__main__':
    n = int(input())
    instructions = [input().split() for _ in range(int(input()))]
    res = parking_system(n, instructions)
    for line in res:
        print(line)
