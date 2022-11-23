import random

exit = False
user_points = 0
computer_points = 0

while exit == False:
    options = ["rock", "paper", "scissors"]
    user_input = input("Choose rock, paper, scissors or exit: ")
    computer_input = random.choice(options)

    if user_input == "exit":
        print(f"your score is {user_points}", f"computer's score is {computer_points}")
        print("Game Ended")
        exit = True

    if user_input == "rock":
        if computer_input == "rock":
            print("Your input is rock")
            print("computer input is rock")
            print("It is a tie!")
            print(" ")
        elif computer_input == "paper":
            print("Your input is rock")
            print("computer input is paper")
            print("computer wins")
            print(" ")
            computer_points += 1
        elif computer_input == "scissors":
            print("Your input is rock")
            print("computer input is scissors")
            print("you win")
            print(" ")
            user_points += 1

    if user_input == "paper":
        if computer_input == "paper":
            print("Your input is paper")
            print("computer input is paper")
            print("It is a tie!")
            print(" ")
        elif computer_input == "scissors":
            print("Your input is paper")
            print("computer input is scissors")
            print("computer wins")
            print(" ")
            computer_points += 1
        elif computer_input == "rock":
            print("Your input is paper")
            print("computer input is rock")
            print("you win")
            print(" ")
            user_points += 1

    if user_input == "scissors":
        if computer_input == "scissors":
            print("Your input is scissors")
            print("computer input is scissors")
            print("It is a tie!")
            print(" ")
        elif computer_input == "rock":
            print("Your input is scissors")
            print("computer input is rock")
            print("computer wins")
            print(" ")
            computer_points += 1
        elif computer_input == "paper":
            print("Your input is scissors")
            print("computer input is paper")
            print("you win")
            print(" ")
            user_points += 1


