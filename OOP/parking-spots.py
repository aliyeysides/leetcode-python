from enum import Enum, auto
from typing import List


class Size(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


SIZES = {
    'Small': Size.SMALL,
    'Medium': Size.MEDIUM,
    'Large': Size.LARGE
}

SIZE_LABELS = {e: n for n, e in SIZES.items()}


class Car:
    def __init__(self, size: str, color: str, brand: str):
        self.size = SIZES[size]
        self.color = color
        self.brand = brand

    def __str__(self) -> str:
        return f"{SIZE_LABELS[self.size]} {self.color} {self.brand}"


class ParkingSpot:
    def __init__(self, size: Size):
        self.parked_car = None
        self.size = SIZES[size]

    def park(self, car: Car) -> bool:
        if not self.parked_car and self.size.value >= car.size.value:
            self.parked_car = car
            return True
        return False

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
    def __init__(self, spots: List[str]):
        self.spots: List[ParkingSpot] = []
        self.__size = len(spots)
        self.__free_spots = len(spots)
        for spot in spots:
            self.spots.append(ParkingSpot(spot))

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


def parking_system(spots: List[str], instructions: List[List[str]]) -> List[str]:
    parking_lot = ParkingLot(spots)
    ans = []
    for instruction in instructions:
        command, *args = instruction
        if command == 'park':
            spot, *car = args
            parking_lot.park(int(spot), Car(*car))
        elif command == 'remove':
            parking_lot.leave(int(args[0]))
        elif command == 'print':
            ans.append(str(parking_lot.spots[int(args[0])]))
        elif command == 'print_free_spots':
            ans.append(str(parking_lot.free_spots()))
    return ans


if __name__ == '__main__':
    spots = input().split()
    instructions = [input().split() for _ in range(int(input()))]
    res = parking_system(spots, instructions)
    for line in res:
        print(line)
