# Probe eval: v5_round2_final

checkpoint: `checkpoints/sft_expanded_v2_best_final.pt`

| band | TMR@2 | SSS | TAS |
| --- | --- | --- | --- |
| control(in-list) | 65% | 0.07 | 0.35 |
| unseen-high-freq | 70% | 0.11 | 0.40 |
| unseen-mid-freq | 88% | 0.06 | 0.46 |
| unseen-oov | 0% | 0.00 | 0.00 |

## Per-topic

| topic | corpus_hits | TMR@2 | mention | SSS | TAS |
| --- | --- | --- | --- | --- | --- |
| bird | 16601 | 60% | 0.67 | 0.12 | 0.39 |
| cake | 2232 | 80% | 0.80 | 0.10 | 0.45 |
| park | 18427 | 80% | 0.60 | 0.06 | 0.33 |
| sharing | 11101 | 40% | 0.47 | 0.02 | 0.24 |
| moon | 463 | 80% | 0.87 | 0.10 | 0.48 |
| school | 1109 | 80% | 0.80 | 0.02 | 0.41 |
| king | 625 | 100% | 1.00 | 0.18 | 0.59 |
| baby | 1155 | 40% | 0.33 | 0.04 | 0.19 |
| farm | 1187 | 40% | 0.47 | 0.16 | 0.31 |
| beach | 1212 | 80% | 0.73 | 0.14 | 0.44 |
| dragon | 431 | 80% | 0.80 | 0.04 | 0.42 |
| robot | 273 | 100% | 1.00 | 0.06 | 0.53 |
| pirate | 273 | 100% | 0.87 | 0.02 | 0.44 |
| train | 594 | 100% | 1.00 | 0.10 | 0.55 |
| rainbow | 706 | 60% | 0.60 | 0.08 | 0.34 |
| zorp | 0 | 0% | 0.00 | 0.00 | 0.00 |

## Generated stories

### bird (control(in-list), corpus_hits=16601)

**sample 0**: Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved to play with. One day, she went to the park with her mom and her ball.
At the park, Lily saw a small, frightened bird on the ground. The bird was scared because it could not fly. Lily wanted to help the bird, so she picked it up and took it to her mom. Her mom told her that the bird needed to be free and happy.
Lily decided to take care of the bird and make it feel better. She gave it food and water. Soon, the bird was not frightened anymore. It was happy and healthy. Lily and the bird became best friends, and they played together

**sample 1**: One day, a little dog named Spot went for a walk. Spot saw a big red alarm. He did not know what it was, but it looked fun. He went closer to the alarm to see what it was.
Spot heard a sound and started to tremble. He was scared. "What is that?" he asked his friend, a bird named Blue. Blue said, "I don't know, let's go see!" They went closer to the alarm.
When they got close, they saw a big box on the ground. They opened the box and found a big, red ball. Spot and Blue played with the ball all day. They were happy and not scared anymore.

**sample 2**: Once upon a time, there was a pretty bird. The bird liked to sing a lot. One day, the bird met a little boy. The boy said, "Hi, bird! I like your songs. Can you teach me how to sing like you?" The bird said, "Yes, I can teach you. Let's sing together."
The boy and the bird started to sing. They were having fun. But then, something unexpected happened. The bird's feathers started to change colors! The boy was surprised and said, "Wow, bird! Your feathers change colors!"
The bird and the boy were very happy. They sang more and more. They had a new friend. And they lived happily ever after.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a big, tough book called a dictionary. Lily liked to learn new words every day.
One day, Lily's friend, Tim, came to play. "What is that in your dictionary?" Tim asked. "I don't know, let's learn together!" Lily said. They sat down and looked at the dictionary.
Lily and Tim tried to learn together. They learned how to read the tough words. They laughed and had fun. Soon, they knew many new words together. And they lived happily ever after.

**sample 4**: Once upon a time, there was a little bird who lived in a birdcage. The bird was fine and small. The birdcage was in a big tree. The bird liked to sing and play with its friends.
One day, the bird saw a big bird. The big bird asked the little bird, "Do you want to be my friend?" The little bird said, "Yes, I want to be your friend."
The bird and the little bird played together. They had fun every day. The bird would sing and the bird would spin. They were very happy. And they were best friends forever.

### cake (control(in-list), corpus_hits=2232)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to eat cake. One day, she saw a big, yummy cake on the table. Lily wanted to eat the cake, but she knew she should ask her mom first.
Lily went to her mom and said, "Mom, can I eat the cake?" Her mom said, "Yes, but first you must share it with your brother, Tim." Lily was happy and went to find Tim.
Lily found Tim and said, "Tim, can I eat the cake with you?" Tim said, "Yes, Lily, you can eat the cake with me." They both ate the cake and had a fun time. Lily learned that sharing is good and makes everyone

**sample 1**: One day, a little girl named Lucy went to her grandma's house. She knocked on the door and a nice lady opened it. The lady had a big bag of cake mix. Lucy was very happy.
"Can I help you make the cake?" Lucy asked the lady. The lady smiled and said, "Yes, you can help me mix the cake." So, Lucy and the lady worked together to make the cake.
When they finished, the cake was ready. But, oh no! The cake was not for Lucy. It was for a cake that was not for her. Lucy felt ashamed. The lady saw Lucy's sad face and hugged her. "It's okay," said the lady. "We still have fun,

**sample 2**: Once upon a time, there was a pretty girl named Lily. She loved to mix things in her kitchen. One day, she decided to mix yummy food.
Lily's mom came into the kitchen and said, "Lily, do you want some beef for your lunch?" Lily said, "Yes, please!" So, her mom gave her a plate of beef and a spoon.
Lily took her plate of beef and started to mix. She mixed and mixed, making a big mess. Her mom laughed and said, "Lily, that's a funny mix. You can't mix milk like that." Lily felt sad, but she learned that it's okay to try new things. From that day on, Lily used her colors to mix and

**sample 3**: Once upon a time, there was a little girl named Lily. She had a big, tough book called a dictionary. Lily liked to learn new words every day.
One day, Lily's mom said, "Lily, do you want to make a yummy cake?" Lily said, "Yes, Mom! I want to make a delicious cake." So, Lily and her mom went to the kitchen.
Lily and her mom started to make the yummy cake. They mixed the cake stuff in a big bowl. When they were done, they put the cake in the oven. When the cake was ready, they took it out of the oven. Lily was very happy she could help her mom make the yummy cake.

**sample 4**: Once upon a time, there was a little girl who was three years old. She was only three years old. One day, she was playing in her garden when she saw something very shiny. She went closer to take a look and discovered a tasty cake! She was so excited that she started picking it up and eating it. 
Suddenly, she heard a voice coming from behind her. It was her mummy! She had been looking for a treat for the little girl so she ran up to her and said, “Mummy, I found a tasty cake! Can I have it?”
Mummy smiled and said, “Of course!”
So the little girl ate the tasty cake and was so happy to have found the tasty cake. She was

### park (control(in-list), corpus_hits=18427)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to play in the park with her friends. One day, she saw a big, red ball near the swings. Lily wanted to play with the ball, but she was scared to go too far.
Lily's mom saw her looking at the ball. She said, "Lily, do you want to play with the ball?" Lily nodded her head and said, "Yes, please!" Her mom gave her a big hug and said, "Go ahead, and you can try to reach the ball."
Lily went to the park and tried to reach the ball. She jumped and jumped, but she could not reach it. Then, she saw a boy named Tim.

**sample 1**: One day, a little dog named Max went to the park with his best friend, a cat named Fluffy. They liked to play and run around together. Max was a competitive dog, and he always wanted to win. Fluffy was fast, but Max was faster.
At the park, they saw a big dog named Max. Max was not happy. Max said, "Let's race to the big dog!" Max barked, "Okay, let's race!" They ran and ran as fast as they could. They were having so much fun!
But then, something unexpected happened. The big dog stopped running. Max said, "Wait, Max! Look!" Max looked around and saw that he was not really fast. He was in love

**sample 2**: Once upon a time, there was a little boy named Tim. Tim loved to play at the park. His favorite thing to do was to swing on the swings. Tim's favorite thing to do was to swing on the swings. He would go higher and higher than ever before.
One sunny day, Tim went to the park with his mom. He was eager to play on the swings. Tim sat on the swing and back and forth. He felt so happy. His mom pushed him higher and higher. Tim was having so much fun.
Suddenly, Tim saw a big dog running towards him. The dog looked very strong. Tim got scared and started to scream. His mom ran to him and hugged him. The dog stopped and looked at

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of toys, but she did not like to organize them. She would say, "I hate when my toys are too messy!"
One day, Lily went to the park with her mom. She saw a big tree with a hidden door. Lily opened the door and found a box. Inside the box, there were many new toys for her to play with. She was so happy!
Lily played with her new toys all day. But then, something unexpected happened. The toys started to talk! They said, "Lily, you are playing with us!" Lily was very surprised. The toys told her that they were a magic toys, and they could grant

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. He would go from one place to another, until he arrived at a bright and sunny spot. He was so excited!
The little train asked the sun, “Where are you going?”
The sun answered, “We are going to the park to play.”
So, the little train set off on his journey. He saw many fun things and played with the other trains. He also saw how happy the sun was!
At the end of his journey, the little train returned home and told his mommy all about the fun he had. She was so glad that the little train had come to the park. She hugged him and said, “I am

### sharing (control(in-list), corpus_hits=11101)

**sample 0**: Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved to play with. One day, she went to the park to play with her ball.
At the park, Lily saw a boy named Tim. Tim was sad because he had no one to play with. Lily wanted to help Tim, so she said, "Let's share my ball and play together!" Tim smiled and said, "Thank you, Lily!"
Lily and Tim played with the ball all day. They took turns and shared the ball. They had so much fun and became good friends. From that day on, Lily and Tim always shared their toys and played together at the park.

**sample 1**: One day, a little dog named Max went to play outside. Max saw a big red alarm. He did not know what it was, but he wanted to find out. Max ran to his friend, a kind cat named Lily.
Max said, "Lily, I saw a big red alarm. It is very hot. Let's go find something to make it cool." Lily was alert and looked at the alarm. "We can use it to stop the hot alarm," said Lily.
They went to find some shade under a tree. They found a big tree and sat down. Max and Lily stayed in the shade until the alarm stopped. When they were done, Max said, "Thank you, Lily, for helping me feel better

**sample 2**: Once upon a time, there was a pretty girl named Lily. She loved to play with her friends. One day, she found a pretty veil in her mom's room. It was soft and had flowers on it. Lily decided to take the veil home.
Lily showed the veil to her mom. Her mom said, "Lily, you can wear this veil when you are done playing. It will make you look nice." Lily was so happy. She put on the veil and went outside to play with her friends.
Lily and her friends had so much fun playing with the pretty veil. They put on it and looked very nice. At the end of the day, everyone was happy. Lily's mom was proud of her for sharing the

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of toys, but she did not like to share them with her friends. One day, her friend Tom came over to play. Tom was very sad because he did not have any toys to play with. Lily saw that Tom was sad and started to do something nice. She shared her toys with Tom and made him happy. Tom said sorry to Lily and they played together with the toys.
The moral of the story is that sharing makes everyone happy. You can make friends and be happy by being kind and sharing with others.

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. He would always look around to explore and find new friends. One day, he felt very happy and decided to go on a journey.
The little train was feeling very spoiled. He wanted to do something more exciting and so he decided to go to a special place. He was ready to explore!
When he arrived, he saw many wonderful places and lots of exciting things. But he also saw how happy his journey was! Everywhere he looked he was enjoying himself.
At the end of his journey, the little train felt very happy. He had found something so special, and he had found a place to take life. From that day onwards, the little

### moon (unseen-high-freq, corpus_hits=463)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim loved to play outside in the sun. One day, he saw a big, round moon in the sky. The moon was very pretty and bright. Tim wanted to touch the moon, but he was too small.
Tim had a friend named Sam. Sam was very good at making things. Sam saw Tim looking at the moon and said, "Tim, you can't touch the moon. It's too far away." Tim was sad, but he wanted to see the moon up close.
Tim had an idea. He asked Sam, "Can you help me touch the moon?" Sam said, "Yes, I can help you." They both tried to reach the moon

**sample 1**: One day, a little star wanted to travel to the moon. The star was very tired because it was a difficult night. It wanted to see the moon up close and see the stars up close.
The star met a nice bug who could talk. The bug said, "Don't go far away and be very tired. You should go home." The star did not listen to the bug. It wanted to see the moon up close and see the moon.
The star tried to travel, but it was too tired. The star fell down and could not see the moon. The bug tried to help, but the star was too tired. The star crashed on the ground and could not see the moon. The bug was sad, and the

**sample 2**: Once upon a time, there was a pretty comet. The comet was very big and bright. It was a successful comet for all the animals in the forest.
One day, the animals saw something bad. They did not know what it was. They all ran away from the comet to be safe. They hid under the trees and said "thank you" to each other for being safe.
Then, something unexpected happened. The comet turned into a big, bright moon! The animals were so happy. They played together in the sky. They were not scared anymore.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a big, tough teddy bear named Mr. Fluffy. Lily always loved to hug Mr. Fluffy. One day, Lily and Mr. Fluffy went to the park to play.
At the park, they saw a big moon in the sky. They sat on the moon and gazed at the stars. Lily said, "Mr. Fluffy, I wish we could touch the moon." Mr. Fluffy looked at the moon again and smiled.
Then, something unexpected happened. A tiny mouse came out of the moon! The mouse was very happy to see Lily and Mr. Fluffy. The mouse wanted to play with them. Lily was surprised but happy. They all played together and

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey to the moon. Along the way, it met a little bird who said, "Where are we going?" 
The little train said, "We're going to the moon!" 
The little bird said, "Let's go!"
They arrived at the moon and the little train said, "Hello! I was so excited to see you! It must be so much fun!" 
The little bird said, "We would love to play there!"
The little train said, "Let's go! Let's go!"
So, the little train and the little bird ran and played on

### school (unseen-high-freq, corpus_hits=1109)

**sample 0**: Once upon a time, there was a little girl named Lily. She was a very independent girl who liked to do things by herself. One day, Lily's mom told her that they were going to school. Lily was very excited and couldn't wait to go.
At school, Lily met a boy named Tim. Tim was also very excited about school. They started to talk about their day. Lily said, "I like school, Tim. I want to learn more." Tim said, "I like school too, Lily. Let's be friends and learn together."
Lily and Tim became good friends. They learned about school and their school. They also learned about their school and their friends. They were very happy to be independent and learn

**sample 1**: One day, a little boy named Tim went to school. He was a bit scared because it was his first day at school. Tim saw a big stage where kids were going to play. He wanted to join them, so he walked up to them and said, "Hi! Can I join your show?"
The other kids smiled and said, "Yes, Tim! We are here to watch you play. We will watch you have fun." Tim was very happy and ready to go on the stage. But when he got on the stage, he started to feel scared. It was very dark and he could not see well.
The other kids saw Tim on the stage and gathered around him. They said, "Don't be scared,

**sample 2**: Once upon a time, there was a little boy named Tim. Tim loved going to school. Every day, he would go to school and learn. On one big day, Tim was ready to go to school. He was very excited.
But when Tim got to school, he saw that it was very icy outside. Tim was so cold! He could not go to school. His mom saw him and said, "Tim, you need to be patient. It is cold outside."
Tim waited and waited. He was so excited that he could go to school. But then, he started to feel sad. He did not want to wait for the cold to come. He wanted to go to school.
Tim's mom saw that he

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of questions about school. Her teacher, Miss Sue, had a special book called a "xicine". Lily loved to learn new things.
One day, Lily's friend, Timmy, came to visit. He wanted to learn something new too. Miss Sue said, "Let's use the x-ray machine machine." Lily and Timmy got big and looked at the machine. They watched as the machine made a funny noise.
After the machine machine machine was done, Miss Sue asked Timmy, "Do you like the x-ray book?" Timmy said, "Yes, I love the book!" Miss Sue smiled and said, "Great job, Timmy

**sample 4**: Once upon a time, there was a little train who loved to go to school. Every day, the train would wake up at the same time as the other trains.
One day, the little train said to the other trains, "I need to go to school." The other trains said, "We will help you. We know you will like to go to school too."
So, the little train and the other trains packed their luggage and set off on the trip. They were all very excited!
At school, the little train learned how to make new friends and how to have fun. The other trains were happy for the little train and the little train.
The moral of the story is that we should all work together and

### king (unseen-high-freq, corpus_hits=625)

**sample 0**: Once upon a time, there was a king who had a big, shiny crown. The king was very proud of his crown. One day, a little girl named Lily came to the king's castle. She saw the king's crown and wanted it.
Lily said, "King, can I have your crown?" The king said, "No, Lily. This is my crown. You cannot have it." Lily was sad and did not know what to do. She wanted the crown so much.
Then, a wise old owl came to the king. The owl said, "Lily, you must share your crown with the king. It is not nice to take things from others." Lily listened to the owl and gave the crown to

**sample 1**: Once upon a time, there was a king who had a big, shiny crown. He wore it every day because it made him very happy. But the king was a bit selfish. He did not like to share his crown with others.
One day, the king was in a big castle with many castles. He was not happy. He wanted all the pretty flowers and big trees to be his friends. The king thought of a plan. He decided to strike his big, shiny crown on a table. The crown fell on the ground outside the castle.
The people in the land saw the king's shiny crown and thought it was pretty too. They all wanted to play with the crown, but they could not play with it. The

**sample 2**: Once upon a time, there was a king who had a very big castle. The king loved his castle very much. He did not want anyone to take it away. One day, a little girl came to the king and said, "Please, can I take your castle?"
The king was very surprised. He thought the little girl would be rude, but she was nice. So, he said, "Okay, you can take my castle, but you must promise not to tell anyone." The little girl promised, and she took the castle.
The king was so happy. He told his friends about the little girl and the big castle. They all said, "Thank you for giving us your castle. It is even more special for

**sample 3**: Once upon a time, there was a king. The king was very kind and had a big smile. He had many friends in the world. They all loved to play together.
One day, the king's friend, a little girl, lost her toy. The king wanted to help. He used his big hands to find the toy. He looked under the bed, in the toy box, and behind the door. Finally, he found the toy on top of a shelf. The little girl was so happy to have her toy back.
The king was very kind and gave the toy back to the little girl. She hugged the king and said, "I miss you so much, thank you!" The king smiled and hugged her back.

**sample 4**: Once upon a time, there was a king. The king was not ordinary. He had a big black crown. The king liked to wear the crown every day.
One day, the king wanted to complete a big job. He had to help people in his land. He asked people for help. The king was very happy to help.
Together, they completed the job of the king. The people in the land were very proud of him. They gave him money and helped him with his work. The king was very happy. He loved his job and his land.

### baby (unseen-high-freq, corpus_hits=1155)

**sample 0**: Once upon a time, there was a little girl named Lily. She had a big, soft cushion that she loved very much. One day, Lily saw a small, frightened bird outside her window. The bird was cold and hungry.
Lily wanted to help the bird, so she went outside. She took a soft blanket and wrapped it around the bird. The bird felt warm and safe. Lily was happy that she could help the bird.
The next day, Lily went outside to play with the bird. She saw that the bird was not frightened anymore. The bird was just hungry. Lily fed the bird some seeds, and they became her friends. They all played together and had lots of fun.

**sample 1**: One day, a baby was playing with his toys. He was having so much fun! Suddenly, he heard his mommy call his name. He quickly ran to her and started to cry. He wanted to make sure his mommy was ok. 
The baby's mommy was in the kitchen. She was making a special dinner. She was picking out lots of yummy food and putting it in the fridge. The baby was so excited to see the food. 
He opened the fridge door and asked his mommy if she could help him. She said yes, but she had something tasty for him too. The little boy was happy to have his mommy there. 
He took a big bite of the yummy food. But he still felt a little

**sample 2**: Once upon a time, there was a pretty girl named Lily. She loved to play with her toys and have fun. One day, her mommy told her they were going on an adventure. Lily was very happy.
Lily and her mommy went to a big hotel. Inside the hotel, they met a friendly dog. The dog was a helper. He could pick up things and bring them to people. The dog wanted to play with Lily and her mommy.
Lily, her mommy, and the dog played together. They had so much fun! The dog would pick up a ball and bring it back to Lily. They all laughed and had a great time. The end.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a big, tough teddy bear named Mr. Bear. Lily always took Mr. Bear with her everywhere she went.
One day, Lily and Mr. Bear went to the park. They saw many fun things to do. They played on the swings, the slide, and the seesaw. Lily and Mr. Bear had so much fun together.
As the sun started to set, Lily and Mr. Bear went home. They were tired but happy. They had a great day at the park. And from that day on, Lily and Mr. Bear always went to the park together.

**sample 4**: Once upon a time, there was a little baby who lived with his mom and dad. The baby was very tiny but his mom and dad were always there to support him. They loved their little baby very much.
One day, the baby's mom and dad took him to a big room where they all thought it was time to play. The baby was very happy to see so many toys. He saw big toys and small toys, and he even found a useful toy car. The baby loved the toy and played with it all day.
As the sun went down, the baby's mom and dad called him for dinner. The baby ate so much food that he couldn't stop talking about the useful toy car. He just knew that his

### farm (unseen-high-freq, corpus_hits=1187)

**sample 0**: Once upon a time, there was a big farm. On the farm, there was a lot of wheat. The wheat was tall and golden. The farmer was very proud of his wheat.
One day, a little boy came to the farm. He saw the wheat and wanted to play with it. The boy tried to pick up the wheat, but it was too heavy. The farmer saw the boy and said, "Do not worry, I will help you."
The farmer tried to lift the wheat, but it was too heavy for him. The boy started to cry. The farmer felt bad for the boy. He said, "I am sorry, but you cannot help me." The boy was sad and went away.
The next

**sample 1**: One day, a little dog named Spot was playing outside. He saw a big red ball and wanted to play with it. But when he tried to pick it up, a mean cat named Whiskers appeared. Whiskers said, "No, you can't play with my ball!"
Spot thought for a moment and then decided to surrender. He said, "I don't want to fight. Let's play with the ball together!" Whiskers thought about it and decided to try playing with Spot.
They played with the ball all day and had lots of fun. Soon, the other animals saw them playing and wanted to join in. A cat ran up to Spot and Whiskers and said, "Can I play with you?" Spot and Whiskers were surprised but

**sample 2**: Once upon a time, there was a farmer named Tom. He had a big red barn. Tom was a very honest farmer. He always took care of his farm.
One day, Tom went to the farm to get some food for his animals. He saw a little cat. The cat was sad because it wanted to come to the farm too. Tom wanted to help the cat.
Tom took the cat to the farm. The cat was very happy. It shared its food with the other animals. The cat wanted to help Tom. It brought food for the cows, pigs, and chickens. Everyone was happy and they all played together.

**sample 3**: Once upon a time, there was a red barn on a big farm. Inside the barn, there were many animals. They were all happy and played together every day.
One hot day, the sun turned dull. The farmer's animals were not happy. They wanted to find a warm place to stay. They all looked high and low, and then something unexpected happened.
In the clouds, there was a big, bright rainbow. The animals were surprised and happy. They ran through the rainbow and played under it. The sun was gone, and they had fun in the rainbow.

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. He would go from one place to another, until he arrived at a bright and sunny spot. He was so excited!
The little train asked the farmer, “Where are I going?”
The farmer answered, “We are going to the forest to explore.”
So, the little train set off on his journey. He saw many different animals and flowers, and he was so excited!
The little train arrived at the forest, and he met a friendly bird. The bird said, “Where are you going, little train?”
The little train replied, “I am going on a journey to explore the world!”
The bird smiled and said,

### beach (unseen-high-freq, corpus_hits=1212)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to play in the sand at the beach. One sunny day, she went to the beach with her mom and dad. They had so much fun playing in the sand.
Lily saw a big, rough rock near the water. She wanted to see what it was. She asked her mom and dad if they could go near it. They said yes, but they had to be careful. Lily was excited to see what the rough rock could do.
Lily and her mom and dad walked closer to the rock. They touched it and it started to move! They were surprised and a little scared. The rock was not a rock at all! It was a big, rough

**sample 1**: One day, a little boy named Tim went to the beach with his mom. The beach was big and had many sand. Tim liked to play in the sand and water. He also liked to watch the big waves. He had never seen the waves before.
Tim saw a crab in the sand. The crab was very impatient. It did not like how the waves got in its way. It tried to turn the waves, but it was not easy. Tim wanted to help the crab.
Tim sat down next to the crab and talked to it. They played together and had lots of fun. The little crab was happy to have a new friend. Tim was happy too because he made a new friend.

**sample 2**: Once upon a time, there was a pretty beach. The sun was shining and the water was blue. The beach was so blue that everyone was happy.
One day, an old man came to the beach. He had a big box with him. The box said "in an order". He opened it and there was something beautiful inside - a beautiful, shining pearl!
The old man took out the pearl and put it in the sand. He said to the beach, "This is the beach for everyone to enjoy." 
The beach was so filled with children so happy. It felt warm and peaceful in the sun. From that day on, the beach was filled with joy and laughter.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of questions about the beach. She asked her mom, "Why is the beach so dull?"
Her mom said, "The sand is old and not good for the beach. It makes the sand slippery and the sand strong. But the beach can be fun and play in the sand, even when it is not hard."
One day, Lily went to the beach with her mom. She saw the sand in the air and thought about the beach. She started to wonder what else she could explore. 
"Mom, can I go play in the sand?" Lily asked.
Her mom smiled and said, "Yes, but stay here and don't

**sample 4**: Once upon a time, there was a little boy who was three years old. His name was Bob and he was very curious. He explored the beach and saw lots of different things.
One day, Bob noticed something hidden in the sand. He walked over and saw a shrimp. It was an amazing sight. Bob wanted to take a closer look and he slowly reached out to touch it. 
But, when Bob touched the shrimp, a big wave was right in front of him and the shrimp was very angry! Bob was scared and he started to cry. 
His mom heard his cries and came to see what was wrong. She saw the angry shrimp and Bob crying and she hugged him. She explained that the shrimp was probably not

### dragon (unseen-mid-freq, corpus_hits=431)

**sample 0**: Once upon a time, there was a big dragon. The dragon was very sad. He had a terrible day. He did not want to play with the other dragons. He just sat by the window and watched them from the window.
One day, a little girl named Lily came to the dragon. She saw that the dragon was sad. She asked, "Why are you sad, Mr. Dragon?" The dragon told her, "I am sad because I have a terrible day. I have a terrible day."
Lily wanted to help the dragon. She went outside and saw a big tree. She said, "I will climb the tree for you, Mr. Dragon!" Lily climbed the tree and sat on a branch. She looked around

**sample 1**: One day, a little dragon was feeling very nervous. He was all alone in a big cave. He did not know where to go. He was scared to find a friend.
A little girl saw the dragon and wanted to help him. She went into the cave and saw the dragon. "Hi, dragon! I am new here. Don't be scared," she said. The dragon looked at her and smiled. "Come out and play," she said.
The little dragon and the little girl played all day. They played with a ball, they played with a ball, and they played with toys. The dragon was not nervous anymore. They were very happy and became best friends.

**sample 2**: Once upon a time, there was a big dragon. The dragon was very sorry. He did not want to hurt anyone. He just wanted to be friends.
One day, a little girl came to the dragon. She was not scared. She said, "Hi, dragon! Let's play!" The dragon looked at her and said, "Okay, but I am big and scary. I am sorry I scared you."
The little girl wanted to help. She said, "Please don't scare me. I am not scared. I want to be friends." The dragon thought about it and said, "Okay, let's be friends."
From that day on, the dragon and the little girl played together. They had lots of

**sample 3**: Once upon a time, there was a big dragon. The dragon was very dangerous. He lived in a cave near the woods. One day, a little boy named Tom went for a walk. He saw the dragon and was scared.
Tom said, "Please, Mr. Dragon, don't eat me. I just want to introduce myself." The dragon looked at Tom and said, "Hello, little boy. I am a big, dangerous dragon. I am very hungry. Please don't eat me."
Tom was not scared. He ate the dragon, and the dragon became smaller and smaller. The dragon said, "Thank you, Tom. Now I am not so dangerous anymore." They became friends and played together every day.

**sample 4**: Once upon a time, there was a brave little boy called Tom. He lived in a big house with his mum and dad. Tom was playing with his toys when he saw a big switch on the wall. He was curious and wanted to find out what it did.
Tom asked his mum: "Mummy, what does the switch do?"
Mum replied: "It turns on and off when you blink, Tom."
Tom was fascinated and he wanted to try. He jumped up and down and tried to catch the switch. But he wasn't very careful and he slipped and fell. Tom began to cry when he realised he had stepped too close to the switch and his body began to slip.
Mum laughed and hugged him. She said

### robot (unseen-mid-freq, corpus_hits=273)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim had a toy robot that he loved very much. The robot was very special to him. It was a reliable robot that could talk and play with Tim.
One day, Tim and his robot went to the park. They played on the swings and the slide. Tim was very happy. But then, Tim's robot started to complain. "I am tired," said the robot. "I want to go home."
Tim wanted to help his robot. He said, "Don't worry, robot. I will help you find your home." They looked around the park and found the robot's home. The robot was very happy and thanked Tim.
From that day on

**sample 1**: One day, a little boy named Tim went to play outside. He saw a big yellow robot. The robot was very strong and could lift things. Tim wanted to be friends with the robot. He said, "Hi, robot! Can we be friends?"
The robot said, "Yes, Tim! I can be friends with you. But first, I need to find a new place to live." Tim was confused. He did not know where to find a tree or a house. He wanted to help the robot.
Tim and the robot started to look for a new home. They found a big tree with a hole in it. The robot said, "This is a good place for a new home." They went inside the hole

**sample 2**: Once upon a time, there was a robot named Robby. Robby was a crazy robot who loved to play and make people laugh. He lived in a big robot house with a big door.
One day, Robby met a little boy named Tim. Tim was lost and scared. Robby said, "Hi Tim, I am Robby the robot. I can help you find your way home." Tim was happy and said, "Thank you, Robby!"
Robby showed Tim the way to open a big door. Tim was surprised by what he saw. Inside, they found Tim's family. Tim's mom put a happy face in Tim's room and made him laugh. Robby and Tim became best friends, and they played and laughed together every day.

**sample 3**: Once upon a time, there was a robot. The robot was very enthusiastic. It liked to play with the kids in the park.
One day, the robot saw the kids playing. It wanted to join them. So it ran to the kids and said, "Hello! I want to join you." The kids were very happy. They played with the robot all day.
The robot had so much fun. It was very big and strong. The robot and the kids were best friends. They played together every day and the robot was always enthusiastic to say "hello" to the kids.

**sample 4**: Once upon a time, there was a robot. The robot was not ordinary. It could talk and sing. The robot had many friends, like the blocks and the dolls. They all lived together in a big house.
One day, the robot wanted to have a party. It wanted to invite all its friends. It asked a question, "Do you want to come to my party?" The toys and blocks said, "Yes, please!" They were very happy.
The robot took the toys and blocks to the big house. They all played and sang together. The robot was very happy. It had a fun party with its friends. The robot's friends liked the party and they all had a great time.

### pirate (unseen-mid-freq, corpus_hits=273)

**sample 0**: Once upon a time, there was a pirate. He had a big hat and a big hat. He liked to sail on the sea. One day, he saw a little boy. The little boy was sad.
The pirate said, "Why are you sad?" The little boy said, "I lost my toy." The pirate wanted to help. They looked for the toy together.
They found the toy under a big rock. The little boy was happy. The pirate was happy too. They became good friends. They played and had fun.

**sample 1**: One day, a pirate named Tom was in his little boat. Tom was a regular pirate who liked to sail his boat on the sea. He had a big hat and a shiny sword. He also had a dog named Spot. Spot was a good dog and always stayed close to Tom and Spot.
One day, Tom and Spot were on the boat in the water. They saw a big boat with a bad man on the side. The bad man wanted to take Tom and Spot away. Tom and Spot were scared, but they wanted to be brave. They thought of a plan to scare the bad man away.
Tom and Spot jumped off the boat and ran towards the bad man. They jumped on him and made him laugh. The bad

**sample 2**: Once upon a time, there was a pirate named Tom. He was a crazy pirate who loved to play tricks on his friends. One day, Tom was playing with his friend, Sam. Tom hid behind a big tree and waited for Sam to come by.
When Sam arrived, Tom jumped out from behind the tree and said, "I found you, Sam!" Sam laughed and said, "You got me, Tom! Let's start a new game!" Tom thought for a moment and then said, "Okay, we can play hide and seek, but this time, we will hide first."
Sam started to look for Tom, but he could not find him. He started to look for a good hiding spot. Then, he saw

**sample 3**: Once upon a time, there was a pirate. He had a big hat and a red hat. He lived on the boat. Every day, he would sail on his boat and look for treasure.
One day, he found a big treasure chest. It was hidden behind a rock. The pirate was very happy. He opened the chest and saw many shiny and pretty things. He took the treasure and took it home.
The pirate showed his treasure to his friends. They were all very happy. They played with the treasure and had lots of fun. The pirate was the happiest pirate in the world.

**sample 4**: Once upon a time, there was a pirate. He had a big hat and a big stick. He lived on a big boat.
One day, the pirate saw a bird. The bird was hurt. The bird was sad. The pirate wanted to help. He took his big stick and used it to cut the bird's wing.
The bird was happy. It could fly again. The pirate was glad. He sailed his boat and said thank you to the bird. The bird smiled and flew away. The pirate sailed away, knowing he had done something good.

### train (unseen-mid-freq, corpus_hits=594)

**sample 0**: Once upon a time, there was a little train. The train loved to go fast. It would go "choo choo" as it went by. The train was very happy when it could go fast.
One day, the train saw a big, dangerous hill. The train wanted to go up the hill. It did not want to get hurt. So, the train went very fast up the hill. The train was very happy and excited.
When the train got to the top of the hill, it saw a big, scary train. The train was so big and scary. The train was scared, but it did not stop going. The train went up the hill very fast. The train could not stop. The train hit the dangerous

**sample 1**: One day, a little train was going down the tracks. The train loved to go fast. It saw a big station far away and said, "I want to ride there!"
A little boy saw the train and said, "Wait! The train will not allow it to ride here. It's not safe!" The boy was sad, but he went to play with his toys.
A big box came near the train. The box was full of gas. The car was going very fast. The little boy did not know about the gas box. He played with the car and the gas box.
The car said, "Stop! The gas box is dangerous! You could have hurt someone!" The little boy stopped and put the gas

**sample 2**: Once upon a time, there was a little train. The train loved to travel on the track. One day, the train saw a big, strong train coming very fast. The strong train said, "Look at that big train! I want to be strong too!"
The big train felt sad, but it wanted to be strong too. So, the train tried to be brave. It went up to the strong train and said, "I will be strong like you!" The big train felt happy and they started to travel together.
As they traveled, they saw many fun things. They saw big trees, pretty flowers, and even a river. By the end of the day, the strong train was not strong, but it did not

**sample 3**: Once upon a time, there was a train. The train was very enthusiastic. It loved to go fast on the tracks. One day, the train met a little boy. The boy did not understand why the train was so happy.
"Hello, train," said the boy. "Why are you so enthusiastic?"
"I am happy because the train is happy," said the train. "I like to go fast on the tracks."
The boy wanted to know why the train was happy. So, he asked the train, "Why are you so happy?"
The train told the boy about its friends. The boy thought about how the train felt. He decided to understood why the train was happy.
The next day, the

**sample 4**: Once upon a time, there was a train. The train was very organized. It helped take care of the train. The train had a friend named Tom. Tom was a boat.
One day, Tom said, "The train is going to send a letter to someone. We will be there soon." The train did not want to go. It wanted to go to a faraway place. This made the train sad.
The next day, the train was ready. It was ready for the letter. Tom said, "Please, train, go to the faraway place." The train went to the faraway place. It was happy. The train sent the letter to a new place. Tom was happy too. They were friends.

### rainbow (unseen-mid-freq, corpus_hits=706)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a beautiful rainbow in the sky. It was so pretty! Lily wanted to touch the rainbow, but she was too small.
Lily had an idea. She would use her toy car to help her reach the rainbow. She pushed her car to the rainbow and said, "Please, rainbow, come down and touch the rainbow!" The rainbow heard Lily and moved down to the ground.
Lily touched the rainbow and it was so pretty. She was so happy. She ran back to her house and told her mom about the rainbow. Her mom smiled and said, "Good job, Lily! You used your toy

**sample 1**: One day, a little dog named Max went for a walk. Max saw a big rainbow in the sky. He wanted to find the end of the rainbow, so he started to walk. He walked and walked, but could not find the end of the rainbow.
Max saw a big tree and a small hole. He heard a sound from the hole. Max went closer and saw a big rainbow in the hole. But, the rainbow was not there before. Max felt sad.
A little girl named Lily saw Max and asked, "Why are you sad?" Max said, "I can't find the end of the rainbow." Lily smiled and said, "I see where the end is!" They looked until they found the end of the

**sample 2**: Once upon a time, there was a pretty garden. In the garden, there was a big tree. The tree had many fruits. The fruits were prunes.
One day, a little boy named Tim came to the garden. He saw the tree and wanted to eat the prunes. He tried to reach for one, but he was too small. The tree was too tall for Tim.
Tim saw a tall man near the tree. He asked the man for help. The tall man reached up and picked the prunes for Tim. Tim ate the prunes and they were very yummy. They were both happy and became good friends.

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of clothes, but she did not like to wear them. One day, her mom told her, "Lily, please listen to me. I have a surprise for you."
Lily did not listen to her mom. She hid her clothes in her room and went to play outside. Later, it was very hot outside. Lily felt the heat on her face, but she did not know why. She thought, "Maybe something is in the shade."
Lily went back inside and told her mom about the surprise. Her mom smiled and said, "I have a surprise for you." The next day, Lily went back to the park, and her mom had

**sample 4**: Once upon a time, there was a little bird who lived in a birdcage. The bird was fine and quiet. The birdcage was his home and he loved his home.
One day, the bird saw something shiny outside. He flew out to get it. As he flew, he saw that the shiny thing was a big, beautiful rainbow. The bird was very happy and he wanted to show it to his friends.
The bird flew back to the birdcage and showed the rainbow to his friends. They all liked the rainbow too. They played under the rainbow and had a fun day. The bird was very glad he found the birdcage and showed it to his friends. They all lived happily ever after.

### zorp (unseen-oov, corpus_hits=0)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim loved to play with his toy cars. One day, he found a big box in his room. He was very excited to see what was inside.
Tim opened the box and found a new toy car. It was red and shiny. He was so happy and wanted to play with it. But then, he saw his mom's makeup on the table. She looked very serious. Tim knew he should not play with the makeup.
Tim decided to play with his other toys. He put on red on his teddy bear and blue on his toy cars. He made them look funny and not serious. Tim's mom was very happy with him. She gave him a big hug

**sample 1**: One day, a little dog named Max went for a walk. Max saw a big red alarm. He did not know what it was, but it looked fun. He went close to the alarm to see what it was.
Max heard a sound and started to tremble. He was scared. "What is that?" he asked his friend, a cat named Luna. Luna said, "I don't know, let's go see!" They went closer to the alarm.
When they got close, they saw a big box on the ground. They opened the box and found a hat inside! Max and Luna put on the hat and suddenly, they were not scared anymore. They laughed and played together, happy with their new hat.

**sample 2**: Once upon a time, there was a little boy named Tim. Tim loved to play outside with his friends. They liked to play catch and run around the park. Tim's favorite thing to do was to run on the soft grass with his friends.
One sunny day, while Tim and his friends were playing, a big dog came up to them. The dog wanted to play too. Everyone was scared of the dog, but Tim and his friends were not scared. They said, "Hi, dog! Do you want to play with us?"
The dog looked at Tim and his friends and thought for a moment. Then, the dog wagged its tail and said, "I want to play too!" The dog's name was Max. Tim

**sample 3**: Once upon a time, there was a little girl named Lily. She had a lot of questions about everything she did. One day, she saw a long line of ants. Lily thought it was fun to watch the ants march. She wanted to learn more about them.
Lily asked her mom, "Can I watch the ants march?" Her mom said, "Yes, but remember to be careful." Lily was very happy. She took her ruler and counted how many ants she could make. She watched the ants march in a line.
Lily was very proud of herself. She thought she was the best at counting. She asked her mom many questions about the ants she saw. Her mom was proud of her too. They went home and had

**sample 4**: Once upon a time, there was a little train who loved to go on journeys. One day, the train set off on a journey to a huge forest. As it went through the forest, it saw lots of animals. But then, the train saw a little girl who was sad.
The little girl was crying. The little girl said, "My ball is stuck in the tree." The train felt bad for the girl. It wanted to help her. The train tried to get the ball for the girl, but it was too high.
Just then, a big, friendly bird came and saw the girl. The bird asked, "Why are you sad?" The girl said, "My ball is stuck in the tree." The bird
