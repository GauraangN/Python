import random

def roll():
    min_val = 1
    max_val = 6
    return random.randint(min_val,max_val)
    
while True:
    players = input("Enter the number of players (2-4): ")
    if players.isdigit():
       players = int(players)
       if 2<= players <=4:
           break
       else:
           print("Must be between 2-4")
    else:
        print("Must be a digit")

Max_score = 50
Player_score = [0 for _ in range (players)] 

while max(Player_score)< Max_score : #Player_score is a list so the loop will run for every element in the list
    for player_idx in range (players):
     print("Player number ",player_idx+1,"'s turn to roll.")
     current_score = 0
     while True:
      Wroll = input("Roll the dice? Y/N: ")
      if(Wroll.lower()!= 'y'):
         break
    
      else:
        rollTrial = roll()
        if(rollTrial == 1):
             print("ooops....you rolled a 1!!!")
             current_score = 0
             break
        else:
             current_score+=rollTrial
             # The roll value os not shown to keep an ambiguity of the score and display it at the end of the game
     Player_score[player_idx] += current_score
     print("Your total score is: ",Player_score[player_idx])
            