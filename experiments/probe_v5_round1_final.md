# Probe eval: v5_round1_final

checkpoint: `checkpoints/sft_expanded_best_final.pt`

| band | TMR@2 | SSS | TAS |
| --- | --- | --- | --- |
| control(in-list) | 60% | 0.05 | 0.35 |
| unseen-high-freq | 63% | 0.11 | 0.38 |
| unseen-mid-freq | 88% | 0.09 | 0.46 |
| unseen-oov | 0% | 0.00 | 0.00 |

## Per-topic

| topic | corpus_hits | TMR@2 | mention | SSS | TAS |
| --- | --- | --- | --- | --- | --- |
| bird | 16601 | 60% | 0.67 | 0.10 | 0.38 |
| cake | 2232 | 80% | 0.80 | 0.02 | 0.41 |
| park | 18427 | 80% | 0.80 | 0.06 | 0.43 |
| sharing | 11101 | 20% | 0.33 | 0.02 | 0.18 |
| moon | 463 | 40% | 0.40 | 0.12 | 0.26 |
| school | 1109 | 40% | 0.53 | 0.08 | 0.31 |
| king | 625 | 100% | 1.00 | 0.12 | 0.56 |
| baby | 1155 | 60% | 0.60 | 0.06 | 0.33 |
| farm | 1187 | 80% | 0.80 | 0.18 | 0.49 |
| beach | 1212 | 60% | 0.60 | 0.10 | 0.35 |
| dragon | 431 | 80% | 0.80 | 0.06 | 0.43 |
| robot | 273 | 80% | 0.73 | 0.02 | 0.38 |
| pirate | 273 | 100% | 0.80 | 0.12 | 0.46 |
| train | 594 | 100% | 1.00 | 0.14 | 0.57 |
| rainbow | 706 | 80% | 0.80 | 0.10 | 0.45 |
| zorp | 0 | 0% | 0.00 | 0.00 | 0.00 |

## Generated stories

### bird (control(in-list), corpus_hits=16601)

**sample 0**: Once upon a time, there was a little girl named Lily. She had a pet cat named Fluffy. Fluffy was a very obedient cat. They played together every day.
One day, Lily and Fluffy went to the park. They saw a big tree. Fluffy wanted to climb the tree. Lily said, "Be careful, Fluffy!" Fluffy climbed up the tree.
Lily looked down and saw Fluffy on the ground. She said, "Fluffy, you are so high up!" Fluffy looked down and meowed. Lily was happy to see Fluffy safe. They played together until it was time to go home.

**sample 1**: One day, a little dog named Max went for a walk. Max saw a big bush. The bush had green leaves. Max liked the bush. He wanted to play with the leaves. He ran fast to the bush.
Max jumped up and started to play with the leaves. He was very happy. But then, Max did something silly. He started to fall! Max tried to catch up, but he was too small.
A big bird saw Max falling. The bird caught Max and put him back in the bush. Max was scared. He tried to shake the bush, but he could not. The bird ate Max. The day was normal again.

**sample 2**: Once upon a time, there was a pretty bird named Bella. She loved to sing and play with her friends. One day, Bella saw a pretty, shiny thing on the ground. She wanted to take it. Bella thought it would be fun to stir it up.
Bella went to her friend, a wise old owl named Mr. Owl. Mr. Owl knew many things. He told Bella, "You should not use that pretty thing. It can hurt you." Bella listened to Mr. Owl and decided to ask her mom first.
Bella told her mom about the pretty thing. Her mom said, "Bella, you are very smart. You know what is safe not to use things that can hurt you." Bella felt happy and safe

**sample 3**: Once upon a time, there was a little girl named Lily. She had a nosy cat named Max. Max liked to look at everything and always wanted to know what was upstairs.
One day, Lily and Max went to the big tree. Lily saw a little bird sitting on a branch. "Hi, bird!" said Lily. "I like your pretty dress!" The bird looked at Lily and said, "Thank you, Lily! I like your dress too."
Lily and Max played in the tree. They watched the bird fly around. Then, they saw a shiny thing on the ground. It was a key! Lily picked up the key and said, "Max, look what we found!" They used the key to open a little

**sample 4**: Once upon a time, there was a little bird who lived in a birdcage. The bird was fine and small. The birdcage was in a big tree. The bird liked to sing and play with its friends.
One day, the bird saw a big bird. The big bird asked the little bird, "Do you want to be my friend?" The little bird said, "Yes, I will be your friend."
The little bird and the big bird played together. They had fun in the sun. They were happy. The fine birdcage was safe with its new friend.

### cake (control(in-list), corpus_hits=2232)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to eat yummy food. One day, she saw a big, yummy cake on the table. She wanted to eat it all by herself.
Lily's mom saw her looking at the cake. She said, "Lily, do you want to share the cake with your brother, Tim?" Lily nodded her head and said, "Yes, Mom! I want to share the yummy cake with Tim."
So, Lily and her mom sat down to eat the yummy cake. They both enjoyed it very much. After they finished eating, they played together and had lots of fun. Lily was happy that she shared the yummy cake with her brother.

**sample 1**: One day, a little boy named Tim went to the park with his mom. The park had a lot of mist. Tim could not see very far. He was a bit scared, but his mom said, "Don't worry, you can feel safe."
Tim saw a big, fake cake in the park. He wanted to eat it, but his mom said, "No, Tim. That cake is not for eating. Let's go play somewhere else."
Tim was sad, but he sat down and played with his toys. He forgot about the cake and had fun with his mom. They played in the park all day, and Tim felt safe with his mom by his side.

**sample 2**: Once upon a time, there was a pretty girl named Lily. She loved to play with her toys. One day, she found a pretty box. The box was shiny and had a note on it. The note said, "Find the big cake in the box."
Lily went to the store with her mom. She saw a big cake on a high shelf. She wanted the cake very much. She tried to reach the cake, but she was too short. Then, she had a clever idea. She would ask her tall friend, Tom, to help her get the cake.
Lily and Tom went to Tom's house. Tom was good at climbing. He climbed up and got the cake for them. Lily was so happy. She

**sample 3**: Once upon a time, there was a little girl named Lily. She had a nosy cat named Max. Max liked to look at everything and touch everything. Lily loved her mom and dad very much.
One day, Max found a big, round thing in the bathroom. He did not know what it was. He thought it was a toy. Max played with the thing and it made a funny noise. Lily heard the noise and came running to see what happened.
Lily saw Max playing with the thing. She was surprised! She did not know that Max did not know it was a big thing. Lily told her mom and dad about the big thing. They all laughed and clapped for Lily. Max felt happy and learned that not all things

**sample 4**: Once upon a time, there was a little girl named Amy. She loved to eat delicious things. One day, Amy's mom made a big cake for her dad's birthday. Amy was very happy.
Amy wanted to help her mom cut the cake. But her mom said, "No, Amy. You need to wait." Amy was sad, but she listened to her mom. She went to play and played with her toys.
The next day, Amy's mom saw how happy she was with the cake. She said, "Amy, you can help me cut the cake. I promise." Amy was happy, and they cut the cake together. Their dad loved the delicious cake, and they all laughed and had a great day.

### park (control(in-list), corpus_hits=18427)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim loved to play at the park. His favorite thing to do was to slide down the big, red slide. One sunny day, Tim went to the park with his mom.
At the park, Tim saw a new friend. It was a girl named Sue. Sue was playing with a toy that looked like a missile. Tim wanted to play with Sue, but he was scared that she might hurt him. Tim's mom saw him and said, "Don't worry, Tim. I will help you."
Tim's mom took him to the missile. She showed him how to use the missile to make a loud noise. Tim was so happy! He and Sue played together

**sample 1**: One day, a little dog named Max went to the park with his owner, Tom. They loved to play and run in the park. Max was very happy. He liked to chew on sticks. Tom was a bit frustrated because Max would not let him play with his stick.
In the park, Max saw a big tree. He wanted to chew on his stick under the tree. But when he tried, he couldn't reach the leaves. Max tried again and again, but he just couldn't chew. He felt sad and frustrated.
Then, Max's friend, a big bird named Sam, saw Max was sad. Sam flew down and saw Max's sad face. Sam flew up and pushed the tree with Max. Under the tree

**sample 2**: Once upon a time, there was a pretty girl named Lily. She loved to play at the park. One day, she went to the park with her mom. The park was big and had many trees. Lily liked to play there.
At the park, Lily saw a big statue. She looked at it closely. She wanted to examine it. She asked her mom, "Can I examine the statue?" Her mom said, "Yes, but be careful."
Lily looked at the statue and smiled. She was happy it was clean and nice. She liked to examine the statue. Then, she played at the park all day.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of toys, but she had a lot of junk in her room. Lily did not like her junk because it was too old and dull. She wanted to have a fun place to play.
One day, Lily's mom said, "Let's introduce you to a new place!" Lily was very excited. She went to the new place and saw many things. She saw a big tree, a blue rock, and a green tree. Lily played with her toys and had a great time.
As she played, Lily saw another girl named Lily. Lily liked Lily and wanted to be her friend. So, they played together and had a lot of fun.

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey of going faster and faster. 
The train went faster and faster until it could only see the town. 
The train stopped at a big park full of tasty snacks. The train was so excited that it decided to take some snacks and go straight there.
The train set down the park and found a big park with lots of tasty treats. It was so happy! The train set up the swings, slides, and even a lake, full of energy and laughter. 
The train was so happy it had gone fast and had lots of fun. From then on, it always went on a journey

### sharing (control(in-list), corpus_hits=11101)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim loved to play with his toy cars. One day, he found a big box in his room. He was very excited to see what was inside.
Tim opened the box and found a toy car. It was a very fast car. He wanted to play with it all day. But then, he saw that the car was broken. Tim was sad and didn't know what to do.
Tim's mom saw him and asked what was wrong. Tim told her about the broken car. His mom said, "Don't worry, we can fix it together." They worked hard and fixed the car. Tim was happy again, and he learned that sharing can make things better

**sample 1**: One day, a little dog named Max went to play outside. Max saw a big red boot. He thought it was dangerous to play with it. Max wanted to show his friend, a cat named Luna.
Max said to Luna, "Look at this dangerous boot I found. Let's play with it!" Luna was scared but she trusted Max. They played with the boot all day. They had so much fun.
But then, something unexpected happened. The dangerous boot started to move! It was not a boot, but a big, friendly turtle! The turtle said, "Hello, I am Tim. I was hiding as a boot too!" Max, Luna, and Tim all became friends and played together every day.

**sample 2**: Once upon a time, there was a pretty girl named Lily. She loved to play with her toys. One day, she found a pretty blouse in her mom's room. It was red and blue, and she liked it very much. She put it on and felt very happy.
Lily's friend, Tim, came over to play. They played with toys and had lots of fun. Then, Tim saw the blouse and wanted to wear it too. He said, "I want a pretty blouse like yours!" But Lily didn't want to share her pretty blouse. They started to fight over the blouse.
After a while, Lily's mom came in and saw the fight. She said, "You both are very nice. I will

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of toys, but she did not like to share them with her friends. One day, her mom told her she could have a new toy if she was good. Lily thought about it and decided to be good.
Lily waited for her mom to bring her a new toy. She saw her friend Tim playing with a ball. Lily wanted to play with Tim, but she did not know how to share her toys with him. She was stubborn and did not like it. Lily tried to be her friend and showed him how to share her toys.
After some time, Lily wanted to play with Tim's toys. She remembered what her mom said about sharing.

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey of going faster and faster. 
The train went faster and faster until it could only see the port. 
The little train was so excited, it wanted to go even faster. It sped down the tracks and saw lots of other trains.
In the end, the little train made it to the port and found a place to rest. There were so many other trains and people waiting to go in the port too. 
The little train was so happy and proud when it arrived. It stayed in the port for a long time, and it enjoyed the journey with all its new friends.

### moon (unseen-high-freq, corpus_hits=463)

**sample 0**: Once upon a time, there was a big, round moon. The moon was very pretty and bright. It lived in the sky with many stars. The moon was very happy.
One day, the moon wanted to play a game. The game was to see who could roll the fastest. The moon was very competitive. It wanted to win the game. The moon tried very hard to roll as fast as the moon.
The moon rolled and rolled. It rolled and rolled. The moon rolled faster and faster. The moon was so far away. The moon was sad. It could not roll anymore. The moon went back to the sky and rolled away. The moon never saw the moon again.

**sample 1**: One day, a little star named Tim was feeling restless. He wanted to learn about the moon. He saw his friend, the moon, and said, "I want to learn about the moon. Can you help me?"
The moon said, "Yes, Tim. I will help you learn about the moon. Let's go to the park and play."
At the park, Tim and the moon played lots of games. They had so much fun. But then, something unexpected happened. A big bird came and took Tim's shiny star away. Tim was sad and the moon was sad too.
The moral of the story is: When you are feeling restless, you should ask for help. It is always good to ask for help

**sample 2**: Once upon a time, there was a pretty comet. The comet liked to travel and see new places. One day, the comet met a big, dangerous mountain. The planet was sad and scared.
The comet said, "Why are you sad, dangerous volcano?" The ice cream said, "I am sad because I am all alone." The comet wanted to help the ice cream.
The comet had an idea. It said, "Let's travel together. I will show you a nice place to be." The ice cream and the comet packed a big pool of water. They traveled and found the perfect place. They were very happy and became best friends.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a big, blue sky that she loved to look at. One day, she saw a slow, green alien. The alien was very nice. Lily thought the alien liked her too.
Lily wanted to be friends with the alien. She said, "Hi! I like your sky. Can we be friends?" The alien smiled and said, "Yes, I want to be friends too. I like to watch the stars come out. Let's play together."
So, Lily and the alien played under the star sky. They watched the stars and laughed together. They became the best of friends. And they were happy.

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey to find a new place to take you to a new world. As it went faster and faster, the train became more and more tired, but it kept going.
Finally, the train reached a big pond. It was so happy to see all the water. The train felt so much better when it saw the fish jump in the water. The train took the fish back to its old home and they became best friends.
From that day on, the train always went on journeys when it went on a journey. It was excited when it found new and exciting places, and it always felt happy. The end

### school (unseen-high-freq, corpus_hits=1109)

**sample 0**: Once upon a time, there was a little girl named Lily. She had a big school called a university. Lily liked to learn new things. One day, she went to school with her mom and dad.
At school, Lily met a boy named Tim. Tim was not nice. He said, "I hate school. It's not fun." Lily felt sad. She wanted to make Tim happy. Lily had an idea. She said to Tim, "Let's play a game. We can play hide and seek." Tim liked the idea. They played hide and seek. Tim was good at hiding. Lily and Tim had fun. They became friends and played at the university every day.

**sample 1**: One day, a little boy named Tim went to school. He saw a big desk in the room. It was an unknown desk. Tim did not know where it came from. He looked around but saw nothing. He was sad.
Tim had an idea. He went to his friend, Sue. He asked her to lend him the desk. Sue said yes. They both liked the desk. They were happy.
Tim and Sue told their teacher about the unknown desk. They asked their teacher to help them. The teacher said, "I will lend you the desk." Tim and Sue were happy. They learned about the unknown desk and had fun with their friends.

**sample 2**: Once upon a time, there was a little boy named Tim. Tim loved to play at school. His teacher said, "Today, we will have a quiz." Tim was excited to try to do the quiz.
The quiz was about colors and shapes. Tim's friend, Sam, had a fake crayon. Sam said, "This crayon is not real, it is fake. It is not real." Tim did not like the fake crayon because it was fake.
Tim and Sam decided to play a game. They tried to number the trees in the classroom. Tim said, "I will number the trees." Sam said, "I will number the trees." They counted the trees together. Tim was happy. He loved to count and number

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of questions about school. Her teacher, Miss Jane, had a quiz at school. Lily wanted to know what she did.
One day, Lily met a new friend named Tim. Tim said, "Hi Lily! Do you want to play with me?" Lily nodded her head and said, "Yes, let's play together!" They played fun games and laughed a lot.
After the quiz, Lily and Tim were very hungry. They went to Miss Jane's house to eat a delicious cake. They had yummy cake and talked about their fun day at school. They were happy to meet Miss Jane and play together.

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey to the park. Along the way, it saw a group of children playing by a lake. The train was very impressed and said, 'Hello! Can I play with you?"
The children were a little scared and said, 'But train! Go ahead. We're going to have so much fun!'
The train was scared, so it ran away as fast as it could. But the children kept following it. Finally, the train took them to the playground. There they found a great playground full of slides and swings.
The children were so happy that the train set them on the journey. They

### king (unseen-high-freq, corpus_hits=625)

**sample 0**: Once upon a time, there was a king. The king had a big, shiny crown. He loved his crown very much. One day, the king wanted to wear his crown to a party. He was very happy.
At the party, the king saw a big, dangerous snake. The snake wanted to eat the king's crown. The king was scared. He did not want the snake to hurt his crown. So, he thought of a plan.
The king used his crown to scare the snake away. The snake was scared and ran away. The king was happy. He wore his crown and danced around the party. The king and his crown had a fun day at the party.

**sample 1**: One day, a king and his people were very hungry. They wanted to eat a big cake. They saw a beautiful crown on a table. The crown was shiny and pretty.
The king said, "Let's wear the crown!" The people were happy and put the crown on their heads. They thought the crown was very special. They played with the crown all day.
But then, a big wind came. The wind blew the crown off the head. The king and his people were sad. They could not eat the yummy cake. They went home with a sad face.

**sample 2**: Once upon a time, there was a king who had a very big castle. The king loved his big castle very much. One day, the king wanted to improve his castle. He wanted to make it even better.
The king asked his helper to make the big castle better. The helper worked very hard. He used pretty colors and made the castle look nice. The king was very happy with his improved castle.
But one day, something unexpected happened. A little mouse jumped out of the castle! The mouse was not good at cleaning. The king was surprised. The mouse helped the king and put him away. The king was very happy. From that day, the king's big castle was the best his new friend.

**sample 3**: Once upon a time, there was a king. The king was very hungry. He wanted to find some food. The king went to a big tree. He saw a long stick on the ground. The stick was an axe. The king picked up the axe.
The king saw a little bird. The bird was also hungry. The king wanted to help the bird. He went to the bird and took the axe. He did not want to hurt the bird.
The king went home. He took a small piece of bread. The king ate the bread. He was happy. The bird was happy too. The king was not hungry anymore.

**sample 4**: Once upon a time, there was a king. The king was not ordinary. He had a big black crown. The king liked to wear the crown all day.
One day, the king wanted to complete a puzzle. The puzzle had many pieces. The king took the pieces out of the black crown. He put them together. The pieces were now on the board.
The people were happy. They clapped and said, "Yay! We completed the puzzle!" The king smiled. He was proud of his king. They lived happily ever after.

### baby (unseen-high-freq, corpus_hits=1155)

**sample 0**: Once upon a time, there was a little baby. The baby had a crib in a big room. The crib was very soft and cozy. The baby loved to sleep in the crib.
One day, the baby's mommy said, "I will allow you to sleep in the crib." The baby was very happy. The mommy gave the baby a soft blanket to lay on. The baby slept and had sweet dreams.
But then, something unexpected happened. The baby's blanket started to move! The baby was surprised. The blanket took the baby to a new place. The new place was a big, soft bed. The baby was not sad anymore. The mommy and the baby played and had fun. They were happy in their new home

**sample 1**: One day, a baby was playing in the park. The baby saw a big tree and wanted to climb it. The baby started to climb the tree. The baby was happy and smiled.
But then, it started to rain. The baby wanted to go home. The baby looked for a place to hide. The baby found a small cave and went in. Inside the cave, the baby found a box. The baby opened the box and saw many toys. The baby was grateful to find the box.
The baby played with the toys and shared them with other kids. The kids took the toys and went home. The baby and the parents were very thankful to the baby for finding the toys and being so kind.

**sample 2**: Once upon a time, there was a pretty girl named Lily. She loved to play with her toys. One day, she found a pretty blouse in her mom's room. It was her mom's blouse. Lily liked the blouse very much.
Lily wanted to show her mom the pretty blouse. She tried to hurry to the kitchen. But her mom was busy. She did not see Lily. Lily put on the pretty blouse. She wanted to show her mom.
Lily ran to the kitchen. She wanted to show her mom the pretty blouse. But she did not see a toy on the floor. Her foot slipped on the toy. She fell down and the blouse broke. Lily was very sad. The pretty blouse was gone.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of toys, but she did not like to organize them. She was always thinking about the day she played with them.
One day, Lily's mom told her it was time to organize her toys. They were almost done. Lily did not want to clean her room. She wanted to keep playing with her toys. She was not happy about this.
Lily's mom mentioned that she had to clean her room before she could play with her toys. But Lily did not want to. She wanted to keep playing with her toys. So, she started to organize her room. She put her cars in one box and her dolls in another box. It was fun

**sample 4**: Once upon a time, there was a little baby who lived with his mom and dad. The baby was very tiny, but he was still very strong. One day, the baby's mom and dad went to buy some food for dinner. They brought their little baby with them.
When they arrived at the store, the baby saw a harmless toy. He wanted it so much! The baby asked his mom and dad if he could have it. They said yes, but he had to be very careful with it. The baby played with the toy all day, but he was happy to have his own toy.
At the end of the day, the baby's mom and dad were tired. They went back to their house and the baby put

### farm (unseen-high-freq, corpus_hits=1187)

**sample 0**: Once upon a time, there was a big farm. On the farm, there was a lot of wheat. The wheat was tall and golden. The farmer liked to take care of the wheat.
One day, a little boy came to the farm. He saw the wheat and wanted to play with it. The boy started to pull the wheat. The wheat was very heavy. The boy pulled and pulled, but the wheat would not move.
The boy did not give up. He pulled harder and harder. Finally, the wheat started to move. The boy was happy. He pulled the wheat all the way to his home. The farmer was happy too. They all lived happily ever after.

**sample 1**: One day, a little boy named Tim went to the farm with his mom. They saw many animals like cows, pigs, and chickens. Tim was very happy to see all the animals. He wanted to take some for his lunch.
When Tim and his mom got to the barn, they saw a big, green patch of corn. It was very impressive! Tim's mom said, "Wow, this is the most impressive corn I have ever seen!"
Tim and his mom decided to take some corn home. They put the corn in a big bag and went back to the car. Tim put the bag in his room and smiled. He knew they would come back to the farm soon.

**sample 2**: Once upon a time, there was a farmer named Tom. He had a big red barn. Tom was a very honest farmer. He always told the truth.
One day, Tom went to his barn. He wanted to get his animals. He saw a small bug. The bug was only an inch long. Tom wanted to keep the bug. He took the bug to his farm.
Tom put the bug on a leaf. The bug was happy. It blinked its eyes and flew away. Tom was happy too. He went back to his big red barn. And he lived happily ever after.

**sample 3**: Once upon a time, there was a big farm. On this farm, there was a big bull. The bull was very sad because he had no food. He was very hungry.
One day, the farmer had to raise a big truck. The truck had to take the goat to the farm. The goat was very happy. He knew he would eat the yummy food from the farm.
The bull got to the farm and started to eat. He did not know that the farm was a bad place. When the farmer came back, he saw that the bull was eating his food. The farmer was very sad. The bull felt bad too. They had to go to the doctor's office. The bull was not happy anymore.

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey of journey to find a new place. As it traveled, it saw lots of different places, like big farms and farms. It was such a weird journey, but the little train kept going. After a long time, the train finally found a beautiful farm. The little train was so happy and excited. It knew it was finally on its journey.

### beach (unseen-high-freq, corpus_hits=1212)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to play in the sand at the beach. One day, she found a big, pretty shell. She was very happy and wanted to show it to her mom.
Lily ran to her mom and said, "Look, Mom! I found a big shell!" Her mom smiled and said, "That's a very nice shell, Lily. But be careful, it might be harsh."
Lily wanted to show the shell to her friends. She put it in her bag and went to the beach. She saw her friends playing in the sand. They were all having fun, but she wanted to show them the shell.
Lily took out the shell and showed it to

**sample 1**: One day, a little boy named Tim went to the beach with his mom. The beach was big and had many sand. Tim liked to build sand castles and play in the water. His mom always had a reliable friend in him to play with him.
Tim's mom said, "Let's add more sand to our sand castle." They added more sand until the sand castle was big and strong. Tim was very happy. He loved playing at the beach with his mom.
But then, a big wave came and washed away their sand castle. Tim was sad and his mom was sad too. They could not decorate the sand castle anymore. The beach was not fun, and they went home with a bad ending to their day.

**sample 2**: Once upon a time, there was a pretty beach. The beach had lots of sand, water, and a big blue sea. The pretty beach was the best place to play.
One day, a little girl named Lily went to the beach with her mom and dad. They had a fun day playing in the sand and swimming in the sea. The sun was shining, and the water felt nice.
As the day went on, Lily and her mom and dad got tired. They decided to sit on the sand and rest. Lily's favorite towel was her favorite. She hugged it tight and smiled. Her mom and dad were happy to have her the pretty beach, and they all had a great time at the pretty beach.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of toys, but she had a lot of junk in her room. Lily did not like her junk because it was too old and dull. She wanted to make her room tidy and nice.
One day, Lily was playing with her junk in the park. She saw a big box in her way. She thought it would be fun to hide her toys in the box. She tried to jump out, but she could not. She was very sad and angry.
A little boy named Tim saw Lily and asked, "Why are you sad?" Lily told him about the box and her junk. Tim said, "I can help you, but you must promise

**sample 4**: Once upon a time, there was a little boy who was three years old. His name was Bob and he was very curious. Bob's mum said she was going to introduce him to a special building.
When Bob arrived, he saw an igloo in the middle of the beach. It was an igloo, and Bob was curious to know what it was. He asked his mum what it was.
Mum said, "It's a unique igloo, Bob. It's full of colourful things." Bob was amazed and couldn't believe it. He wanted to go inside the igloo and learn about it.
So Bob and his mum went inside the igloo. Bob had so much fun learning about the unique igloo. He couldn't wait to come back

### dragon (unseen-mid-freq, corpus_hits=431)

**sample 0**: Once upon a time, there was a big dragon. The dragon was very sad. He was sad because he had no friends. One day, he saw a little girl. She was playing with a ball. The dragon wanted to play too.
The dragon went to the girl and said, "Can I play with you?" The girl looked at the dragon and said, "Yes, you can play with me." They played with the ball and had fun. The dragon was not sad anymore.
Then, something unexpected happened. The dragon started to grow bigger and bigger. He was not a dragon anymore. He was a big dragon! The girl was surprised. The dragon said, "I am a dragon. I was a dragon, but

**sample 1**: One day, a little dragon was feeling very nervous. He was all alone in a big cave. He did not know where to go. He was scared, but he knew he had to find a new home.
The dragon met a big dragon. The big dragon said, "Why are you nervous, little dragon?" The little dragon said, "I don't know how to find a new home." The big dragon said, "I will help you!" They looked for a new cave, but they could not find one.
Then, something unexpected happened. The little dragon found a big, shiny rock. The big dragon was happy and said, "This is my new home!" The little dragon was not nervous anymore. They both lived

**sample 2**: Once upon a time, there was a big dragon. The dragon was very sorry. He had a friend, a small bird. They lived in a big cave. The bird could fly high and fast. The dragon liked to watch his friend fly.
One day, the dragon had to send a letter. The letter was from the dragon. The letter was from a big, mean dragon. The big, mean dragon wanted the letter. The dragon tried to take it. The little bird was scared.
The little bird had an idea. She sang a song. The song made the big, mean dragon dizzy. The dragon flew away. The dragon was happy. He could now get the letter. The little bird and the big, mean dragon

**sample 3**: Once upon a time, there was a big dragon. The dragon was very dangerous. He lived in a cave near the woods. One day, a little boy came to the cave. He wanted to play with the dragon.
The dragon saw the boy and wanted to play too. The dragon said, "Let's play! I am dangerous. You cannot play with me." The little boy did not listen. He just laughed and said, "You can't play with me!" The dragon was sad. He wanted to play with the boy.
The dragon tried to play with the boy, but he was not careful. He broke a big rock. The dragon fell into the lake. The little boy swam to the dragon. The dragon was

**sample 4**: Once upon a time, there was a brave little boy named Tom. He lived in a big house with his mom, dad, and his dog, Max. Tom had a special power that made him strong and happy.
One day, Tom and Max went for a walk. They saw a big tree with a door. Tom said, "Max, let's see what's inside!" They opened the door and found a room full of toys. Tom and Max played with the toys and had lots of fun.
After a while, Tom said, "I am tired. Let's go home and rest." Max barked, "Woof! Woof!" They walked back to their house, tired but happy. Tom and Max slept well, dreaming of

### robot (unseen-mid-freq, corpus_hits=273)

**sample 0**: Once upon a time, there was a robot. The robot was very big and strong. It could lift heavy things. The robot had a friend named Tim. Tim was a little boy who liked to play with the robot.
One day, Tim and the robot went to the park. They played on the swings and the slide. They had lots of fun. But then, the robot started to feel tired. It did not want to play anymore.
The robot tried to lift a big box, but it was too heavy. The robot fell down and the box broke. Tim was sad. The robot was sad too. They could not play anymore. The end.

**sample 1**: One day, a little boy named Tim went to play outside. He saw a big red car. It was an enormous car! Tim wanted to go for a ride. He got in the car and started to drive. He had so much fun driving around.
As Tim drove, he saw a big tree. The tree had a lot of lumber. The lumber was for the old man who lived in the forest. Tim thought it would be fun to play with the tree.
Tim played with the lumber all day. He made a big pile of lumber. He hit the lumber with his hands. The lumber made a loud noise. Tim was very happy. At the end of the day, Tim took the enormous red car home. He had

**sample 2**: Once upon a time, there was a robot named Robby. Robby was very flexible. He could bend and twist his body in many ways. Robby had a friend named Tim. Tim was a little boy who loved to play with Robby.
One day, Tim and Robby went to the park. Tim saw a big tree and wanted to climb it. He asked Robby, "Can I climb the tree?" Robby said, "Yes, you can." Tim climbed the tree while Robby jumped up and down.
Tim and Robby had so much fun at the park. They played on the swings and went down the slide. They laughed and played all day. Tommy was happy to have a flexible robot friend like Robby to play with.

**sample 3**: Once upon a time, there was a robot. The robot was very enthusiastic. It liked to play with the kids in the park.
One day, the robot saw the kids playing. The robot wanted to join them. It went to the kids and said, "Can I play with you?" The kids were happy and said, "Yes, you can play with us!"
The robot and the kids played all day. They had so much fun. The robot was happy it went to the park. It made new friends and they all played together.

**sample 4**: Once upon a time, there was a robot. The robot was not big or fancy. It was very small and had many legs. The robot had a friend named Tim. Tim was not too big or too small. He was just a robot.
One day, the robot asked Tim, "Do you want to go to a party?" Tim said, "Yes, I want to go!" The robot was happy. They went to the party and had fun. They ate cake and danced. The robot and Tim were very happy.
Then, something unexpected happened. The robot started to grow bigger and bigger. It was a magic robot! The robot said, "Thank you, Tim, for the best party ever!" Tim was surprised,

### pirate (unseen-mid-freq, corpus_hits=273)

**sample 0**: Once upon a time, there was a pirate named Tom. Tom had a big hat and a big ship. He liked to sail on the sea with his friends.
One day, Tom and his friends found a treasure chest. They were very happy. Tom said, "Let's open the chest and see what's inside!" They all agreed and started to open the chest.
Inside the chest, they found many toys and games. They played and had fun all day. Tom and his friends were very happy. They knew that the treasure was a special gift from the pirate.

**sample 1**: One day, a pirate named Tom was in the water. He was a big pirate who liked to pinch a little boy named Tim. Tim was very scared of the pirate.
Tom saw Tim and said, "Why are you scared, little pirate?" Tim said, "I have a big pirate friend and a little boy. Will you help me?" Tom said, "Yes, I will help you."
Tom used his big pirate hands to pinch Tim's hand. Tim cried out, "Ouch! That hurt!" Tom felt bad. He said, "I'm sorry, little pirate. I won't pinch you anymore."
The moral of the story is: don't pinch someone is not nice. It can hurt and make you sad

**sample 2**: Once upon a time, there was a pirate named Tom. He was a crazy pirate who loved to play tricks on his friends. One day, Tom was playing with his friend Sam when a big ring came from his boat.
"Hello, Tom!" said Sam. "Can I play with you?" Tom thought for a moment and said, "Okay, but you must be careful. The ring is very big and will scare you if you don't stop."
Sam agreed and took a little spray bottle. He sprayed the big bell on the ground, just like Tom said. The bell made a loud noise, and Sam got scared.
"Tom, stop it! You are hurting yourself!" Sam shouted. Tom felt sad and said,

**sample 3**: Once upon a time, there was a pirate named Tom. He was a very fit pirate who liked to sail on the big sea. Tom had a friend named Sam. Sam was a small boy with a big hat. They liked to play and have fun together.
One day, Tom and Sam found a big, shiny treasure chest. They were very excited and wanted to see what was inside. They tried to open it, but it was too heavy. Tom tried to help Sam, but he could not. They opened the chest and found many toys inside.
Tom and Sam thought they would never be friends with the treasure. But then, they saw a big pirate from the chest. The pirate was not mean, he just wanted to play

**sample 4**: Once upon a time, there was a pirate named Tom. Tom was a troubled pirate. He lived on a big ship with his friends on the ship. They liked to look for gold and animals.
One day, Tom saw a box on the ship. He said, "What's this?" Tom tried to remove the box from the ship. It was very heavy. Tom pulled and pulled. The box started to move.
Tom and his friends found more gold and jewels. They were very happy. They said, "Yay, we found gold!" They took the treasure and went back to the ship. Tom and his friends played and had fun all day.

### train (unseen-mid-freq, corpus_hits=594)

**sample 0**: Once upon a time, there was a little train. The train loved to go fast on the tracks. It would go "choo choo" as it went by. The train was very happy when it could go fast.
One day, the train saw a big, dangerous hill. The train was scared, but it wanted to go up the hill. So, it started to go up, up, up the hill. The train went faster and faster, and it was so much fun!
At the top of the hill, the train saw a big, beautiful rainbow. The rainbow was so pretty, and it made the train feel happy. The train knew that it was safe up there, and it was not scared anymore. The train went

**sample 1**: One day, a little train was going down a long track. The train was very tired. It saw a big station far away and said, "I want to stop and rest."
A nice man came to the station. He saw the tired train and said, "Do not worry, little train. I will help you stop." The man picked up the train and put it in a safe place. The train stopped and said, "Thank you, nice man."
The train went home and sat on the track. The train was happy and started to rest. The next day, the train came back to the station. The train said, "Thank you, nice man, for helping me rest. I have train now." The man

**sample 2**: Once upon a time, there was a little train. The train liked to go fast. It lived near a big, old train station. The train was very proud of its speed.
One day, the train met a little boy. The boy was sad. The train asked, "Why are you sad?" The boy said, "I lost my toy. It is an original toy, not like the train's original one."
The train wanted to help. It said, "I will find your toy. But first, I need to go fast. I have an order. Can you delay your trip?" The boy said, "I will wait for you, train."
The train was excited. It went fast and stopped at the

**sample 3**: Once upon a time, there was a train. The train was very fast. It liked to go fast on the tracks. One day, the train met a slow train. The slow train did not have much time.
The slow train said, "Hi, little train! Do you want to go fast with me?" The slow train said, "Yes, I want to go fast with you." So they went fast together. The slow train was happy.
But one day, the slow train stopped. The slow train was sad. It could not go fast anymore. The slow train said, "Don't be sad, you can go fast with me." The slow train felt better. They went fast and had fun. The slow train

**sample 4**: Once upon a time, there was a train. The train was very organized. It helped take care of the train. The train had a friend named Tim. Tim was a boy.
One day, Tim asked the train, "Can you help me increase my strength?" The train said, "Yes, I can help you." So, the train let Tim ride on it.
The train went up, down, down. It went slow and steady. Tim was happy. They were both very good at shrinking things. And they lived happily ever after.

### rainbow (unseen-mid-freq, corpus_hits=706)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big rainbow in the sky. The rainbow was very pretty. Lily wanted to find the end of the rainbow.
Lily asked her friend Tom, "Do you know where the end of the rainbow is?" Tom said, "I don't know, but let's go look!" They walked and walked until they found a big tree. The tree was very tall and had many leaves.
Lily and Tom climbed the tree. They saw a big, dead tree. They were scared, but they wanted to see what was inside. They climbed down the tree and found a big, beautiful rainbow. The rainbow was

**sample 1**: One day, a little dog named Max went for a jog in the park. The sun was shining, and the birds were singing. Max loved to run and play with his friends. As Max jogged, he saw a big rainbow in the sky. He was very happy and wanted to show his friends the rainbow.
Max jogged back to his house with the rainbow. But as he got closer, he saw that the rainbow was not a real rainbow. It was made of big, round, and colorful balls. Max was sad and wanted to find another rainbow.
He jogged further and farther through the park. Suddenly, Max saw a big box. It was not a rainbow, but a big, round, colorful ball. Max was so happy

**sample 2**: Once upon a time, there was a pretty garden. In the garden, there were many pretty flowers. One day, a little girl named Lily came to the garden.
Lily saw a big, pretty rainbow. The rainbow was very important. It made the rainbow look like a big rainbow. Lily wanted to share the rainbow with her friends.
Lily took the rainbow to her friends. They were all happy to see the pretty rainbow. They played together and had a fun day. Lily was happy she could share the important rainbow with her friends.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of toys, but she had a lot of junk in her room. Lily did not like her junk because it was too old and dull. She wanted to make her room look nice again.
One day, Lily was playing with her junk. She made a big mess in her room. There were cars, dolls, and balls. Lily was very happy with her work. She thought her junk was perfect.
But then, something unexpected happened. While Lily was playing, her mom came in her room. She saw the junk and got very angry. She told Lily that she could not play with her junk anymore. Lily felt sad and her mom helped her clean

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey of journey to find a new friend. As it traveled, it saw something amazing - a big rainbow in the sky!
The little train was so happy and excited that it raced up to the rainbow. But as it got closer, the rainbow disappeared. The little train was sad and confused.
Suddenly, a rainbow appeared in the sky and disappeared again! The little train was so surprised - it had never seen a rainbow before!
The little train asked the rainbow, "Where are you?"
The rainbow replied, "I am here to help."
The little train was so happy and asked the rainbow

### zorp (unseen-oov, corpus_hits=0)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim loved to play with his toy box. One day, he found a big box in his room. He was very happy and wanted to see what was inside.
Tim opened the box and saw a small, frightened bird. The bird was scared and could not fly. Tim wanted to help the bird. He thought of a plan to make the bird feel better.
Tim took the bird outside and let it go. The bird started to fly again. But then, something unexpected happened. The bird turned into a big, friendly dog! Tim was so surprised. The dog wagged its tail and licked Tim's face. They became best friends and played together every day.

**sample 1**: One day, a little dog named Max went to play. Max saw a big ball and wanted to play with it. He ran and ran to get the ball. But a mean cat named Tom saw Max and wanted the ball too.
Tom had a big eye that could see very far. "I want the ball!" Tom said. Max did not want to give the ball to Tom. They both yelled, "I want the ball!"
A bird heard them and flew down to help. The bird said, "You both can have the ball if you share." Max and Tom took turns playing with the ball. They had a lot of fun together.
In the end, Max and Tom learned that sharing is good. They became best

**sample 2**: Once upon a time, there was a little boy named Tim. Tim loved to play outside. One day, it was so gloomy outside. Tim wanted to go to the park to play. He put on his coat and hat, and took his camera.
At the park, Tim saw a big tree. He wanted to take a picture of the tree. Tim took a big, red ball and drew the tree on a page. The tree looked very nice.
After playing, Tim was tired. He took out his camera and took a big picture of the tree. Then, he took a picture of the tree. Tim was very happy with his picture. He went home, smiling at the gloomy day, thinking about the fun day at the

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of toys, but she had a lot of junk in her room. Lily did not like her junk because it was too old and dull. She wanted to make her room tidy and nice.
One day, Lily was playing with her junk in her room. She found a big box under her bed. She thought it would be fun to hide her toys in the box. She put the toys in the box and smiled. She thought it was a good hiding place for them.
But when her mom came into the room, she saw the big box and the junk. She told Lily that the junk was not a toy. It was her dad's junk to

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey of adventure. It was very sunny and bright.
The train went through the dark forest, through the tall grass and the little hills. It saw many pretty flowers and trees. The train was having a good time.
When the train finally found a nice spot, it made a nice warm place to rest. The train was happy it had been on the journey of adventure. The end.
