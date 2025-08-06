# Curling-Game-Simulator
This is a simple 2D top-down curling game simulator. Made with pygame.<br><br>

### Rules and Game Logic
* The game lasts eight ends, with extra ends in the event of a tie.
* Scoring is like normal curling, counting only rocks that are in the house and only the closest rocks to the pin until the colour changes. (e.g. if Red has the 1st, 2nd, and 4th closest rocks to the pin but Yellow has the 3rd closest, Red will score 2 points)
* On game start, the ice conditions are randomized so the player(s) will not know how fast/slow or straight/curly the ice will be until a rock has been thrown.
* The starting colour is also randomized by a 50/50 coin flip on game start.
* The starting colour after the first end is determined by which colour scored in the previous end. (e.g. if Red scored in the previous end, they will go first in the current end)
* If the thrown rock stops before or on the far hog line, it will be taken out of play unless it makes contact with another rock first.
* If ANY rock touches the sides of the sheet (top/bottom from the player's perspective) or completely passes the back line at ANY time, that rock will be taken out of play.
* The Free Guard Zone (FGZ) is in effect, meaning that any rock that is in play, but outside of the house, cannot be removed from play until after the 5th rock. If an FGZ rock is taken out, the thrown rock will instead by taken out of play and the rocks will be reset to their previous arrangement. Essentially the result is just that the colour of the thrown rock will lose a shot.
* The rock physics are designed to mimic real life as closely as possible, so collisions should act approximately the way you would expect.

### Controls
* W - Increase clockwise rotation<br>
* S - Increase counterclockwise rotation<br>
* LEFT - Aim left<br>
* RIGHT - Aim right<br>
* UP - Increase power<br>
* DOWN - Decrease power<br>
* SPACE - Confirm
