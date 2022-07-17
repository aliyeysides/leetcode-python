from enum import Enum, auto


class Card:
    @property
    def card_value(self) -> int:
        raise NotImplementedError()

    def __lt__(self, other):
        return self.card_value < other.card_value


class Suit(Enum):
    CLUBS = auto()
    DIAMONDS = auto()
    HEARTS = auto()
    SPADES = auto()


class PlayingCard(Card):
    SUITS = {
        "Clubs": Suit.CLUBS,
        "Diamonds": Suit.DIAMONDS,
        "Hearts": Suit.HEARTS,
        "Spades": Suit.SPADES,
    }
    SUIT_NAMES = {e: n for n, e in SUITS.items()}
    VALUES = {
        "A": 1,
        **{str(i): i for i in range(2, 11)},
        "J": 11,
        "Q": 12,
        "K": 13,
    }
    VALUE_NAMES = {e: n for n, e in VALUES.items()}

    def __init__(self, suit: str, value: str):
        super().__init__()
        self.__suit = self.SUITS[suit]
        self.__value = self.VALUES[value]

    @property
    def card_value(self) -> int:
        return self.__value

    def __str__(self) -> str:
        value = self.VALUE_NAMES[self.__value]
        suit = self.SUIT_NAMES[self.__suit]
        return f"{value} of {suit}"


class Color(Enum):
    RED = auto()
    BLACK = auto()


class JokerCard(Card):
    COLORS = {
        'Red': Color.RED,
        'Black': Color.BLACK
    }
    COLOR_NAMES = {e: n for n, e in COLORS.items()}

    def __init__(self, color: str):
        super().__init__()
        self.__color = self.COLORS[color]
        self.__value = 14

    @property
    def card_value(self) -> int:
        return self.__value

    def __str__(self) -> str:
        color = self.COLOR_NAMES[self.__color]
        return f"{color} Joker"


class Hand:
    def __init__(self, cards: list[Card]):
        self.cards = [*cards]

    def __str__(self) -> str:
        return ", ".join(str(card) for card in self.cards)

    def __lt__(self, other):
        for card_a, card_b in zip(reversed(sorted(self.cards)), reversed(sorted(other.cards))):
            if card_a < card_b:
                return True
            elif card_b < card_a:
                return False
        return False


class Game:
    def __init__(self):
        self.__deck: list[Card] = []
        self.__hands: list[Hand] = []

    def add_card(self, suit: str, value: str) -> None:
        self.__deck.append(PlayingCard(suit, value))

    def card_string(self, card: int) -> str:
        return str(self.__deck[card])

    def card_beats(self, card_a: int, card_b: int) -> bool:
        return self.__deck[card_a] > self.__deck[card_b]

    def add_joker(self, color: str) -> None:
        self.__deck.append(JokerCard(color))

    def add_hand(self, card_indices: list[int]) -> None:
        self.__hands.append(Hand([self.__deck[i] for i in card_indices]))

    def hand_string(self, hand: int) -> str:
        return str(self.__hands[hand])

    def hand_beats(self, hand_a: int, hand_b: int) -> bool:
        return self.__hands[hand_a] > self.__hands[hand_b]


if __name__ == '__main__':
    game = Game()
    hand_a_list = []
    n_1 = int(input())
    for i in range(n_1):
        suit, value = input().split()
        game.add_joker(value) if suit == "Joker" else game.add_card(
            suit, value)
        hand_a_list.append(i)
    game.add_hand(hand_a_list)
    print(game.hand_string(0))
    hand_b_list = []
    n_2 = int(input())
    for i in range(n_1, n_1 + n_2):
        suit, value = input().split()
        game.add_joker(value) if suit == "Joker" else game.add_card(
            suit, value)
        hand_b_list.append(i)
    game.add_hand(hand_b_list)
    print(game.hand_string(1))
    print("true" if game.hand_beats(0, 1) else "false")
