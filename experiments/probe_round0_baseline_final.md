# Probe eval: round0_baseline_final

checkpoint: `checkpoints/sft_response_best_final.pt`

| band | TMR@2 | SSS | TAS |
| --- | --- | --- | --- |
| control(in-list) | 58% | 0.08 | 0.32 |
| unseen-high-freq | 78% | 0.11 | 0.43 |
| unseen-mid-freq | 80% | 0.09 | 0.44 |
| unseen-oov | 0% | 0.00 | 0.00 |

## Per-topic

| topic | corpus_hits | TMR@2 | mention | SSS | TAS |
| --- | --- | --- | --- | --- | --- |
| bird | 16601 | 100% | 0.89 | 0.20 | 0.54 |
| cake | 2183 | 67% | 0.67 | 0.03 | 0.35 |
| park | 18427 | 33% | 0.44 | 0.00 | 0.22 |
| sharing | 6324 | 33% | 0.22 | 0.07 | 0.14 |
| moon | 463 | 67% | 0.67 | 0.07 | 0.37 |
| school | 1109 | 33% | 0.44 | 0.00 | 0.22 |
| king | 625 | 100% | 1.00 | 0.13 | 0.57 |
| baby | 1155 | 100% | 1.00 | 0.03 | 0.52 |
| farm | 1187 | 100% | 0.67 | 0.27 | 0.47 |
| beach | 1212 | 67% | 0.67 | 0.17 | 0.42 |
| dragon | 431 | 100% | 1.00 | 0.10 | 0.55 |
| robot | 273 | 33% | 0.33 | 0.00 | 0.17 |
| pirate | 177 | 100% | 1.00 | 0.17 | 0.58 |
| train | 594 | 100% | 1.00 | 0.10 | 0.55 |
| rainbow | 706 | 67% | 0.67 | 0.07 | 0.37 |
| zorp | 0 | 0% | 0.00 | 0.00 | 0.00 |

## Generated stories

### bird (control(in-list), corpus_hits=16601)

**sample 0**: Once upon a time, there was a little bird named Tim. Tim was a very foolish bird. He liked to play and fly all day long. One day, Tim saw a big tree with a lot of fruit. He wanted to eat the fruit, but he was too small to reach it.
Tim had an idea. He would ask his friend, the tall giraffe, to help him. The giraffe was very tall and could reach the fruit. Tim asked the giraffe, "Can you help me get the fruit?" The giraffe said, "Yes, I can help you."
The giraffe reached up and picked the fruit for Tim. But when the giraffe gave the fruit to Tim, it was not a fruit at all! It was a

**sample 1**: One day, a little dog named Max went for a jog in the park. The sun was shining, and the birds were singing. Max loved to run and play with his friends. As Max jogged, he saw a little bird on the ground. The bird looked hurt and sad.
Max stopped and said, "Little bird, why are you hurt?" The bird said, "I fell from my nest and now my wing hurts." Max knew he had to help the bird. He picked up the bird and carried it to his home.
Max took care of the bird until it was better. They played in the park together and became best friends. The bird's wing was all better, and they were both very happy.

**sample 2**: Once upon a time, there was a pretty princess. She lived in a big, expensive castle. The princess had a mommy and a daddy. She liked to play with her toys and laugh all day.
One day, the princess went outside to play. She saw a small bird. The bird looked sad. The princess wanted to help the bird. She kneeled down and let the bird sit on her palm.
"Hi, little bird," said the princess. "Why are you sad?" The bird looked at her and said, "I lost my friend. Can you help me find her?" The princess looked around and found the bird's friend. The friend was happy, too. The bird and her friend lived happily in the expensive castle

### cake (control(in-list), corpus_hits=2183)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to eat cake. One day, she saw a big, yummy cake on the table. She wanted to eat it all by herself.
Lily's mom saw her looking at the cake. She said, "Lily, do you want to share the cake with your friends?" Lily thought about it and said, "Yes, Mom, I will share the cake with them."
So, Lily and her mom went to her friends' houses. They all sat down and ate the yummy cake together. They were very happy and full. Lily learned that sharing with others can make everyone happy.

**sample 1**: One day, a little dog named Max went to play. Max saw a big cake on a table. It was an oven. Max wanted to eat the cake. He started to complain. He said, "I want to eat the cake!"
Max's friend, a small cat named Lily, heard him. She came to see what was making the cake. Lily said, "Max, be careful! The oven is hot. We should not play near it."
Max listened to Lily and did not play near the oven. They played together and had fun. They learned that it was fun to play with friends, but they should be careful too.

**sample 2**: Once upon a time, there was a pretty princess named Lily. She lived in a big castle with her mom and dad. Lily loved to play with her toys and make new friends.
One day, Lily went to the kitchen to get a snack. She saw a big ice cream truck. She wanted to go to the ice cream truck to get a yummy treat. She put on her hat and her coat.
Lily walked to the ice cream truck. She saw a nice lady inside. The lady was making ice cream. Lily asked, "Can I have a treat, please?" The lady smiled and gave Lily a tasty ice cream.
Lily was very happy. She sat down and enjoyed her ice cream. She shared it with her mom

### park (control(in-list), corpus_hits=18427)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to play in the park with her friends. One day, she saw a big, red ball. She wanted to play with it, but it was too high for her to reach.
Lily had an idea. She asked her friend, Tom, to help her. Tom was very tall and could reach the ball. He gave it to Lily and said, "Here you go, I will help you." Lily was very happy and said, "Thank you, Tom!"
Lily and Tom played with the big, red ball all day. They had so much fun. When it was time to go home, they said goodbye and went to their homes. Lily couldn't

**sample 1**: One day, a little dog named Max went to the park. Max saw a big red alarm. He did not know what it was. Max was a bit frightened. He thought the alarm would make him feel better.
Max met a big cat named Tom. Tom was sitting on a seat near the alarm. Max said, "Tom, do not play with the alarm. It is not a toy." Tom looked at Max and said, "Oh, I did not know. I am sorry."
But then, something unexpected happened. The alarm started to make a loud noise. Suddenly, the alarm went off! Max and Tom were both surprised. They looked around and saw that the alarm was not a real alarm. It was a toy

**sample 2**: Once upon a time, there was a pretty girl named Lily. She loved to play in the park. One day, she saw a pretty bird. The bird was on a tree. Lily wanted to play with the bird.
Lily said, "Hello, bird! Do you want to play with me?" The bird looked at her and said, "Yes, I want to play. But I am too high up in the tree. Can you help me find a way?"
Lily thought for a moment. She wanted to help the bird. She had an idea. She climbed the tree and picked the bird. She gave the bird a gentle hug. The bird was happy.
Lily and the bird played in the park all day. They

### sharing (control(in-list), corpus_hits=6324)

**sample 0**: Once upon a time, there was a little girl named Lily. She had a big, soft pillow that she loved very much. Every night, she would hug her pillow tight before she went to sleep.
One day, Lily's friend, Tim, came over to play. They played with toys and had lots of fun. But then, Tim saw Lily's pillow and wanted it for himself. He tried to take it away from her. Lily was sad and didn't know what to do.
Lily thought for a moment and decided to share her pillow with Tim. She said, "Let's share the pillow, and we can both be happy." Tim smiled and they both sat down to share the pillow. They played together, and Lily

**sample 1**: One day, a little dog named Max went to play outside. Max saw a big red boot. He thought it was dangerous to play with it. Max wanted to show his friend, a cat named Luna.
Max said to Luna, "Look at this dangerous boot I found. Let's play with it!" Luna was scared but she trusted Max. They played with the boot all day. They had so much fun.
But then, something unexpected happened. The dangerous boot started to move! It was not a boot, but a big, friendly turtle. The turtle said, "Hello, I am Tina the turtle! I live on the boot. Thank you for finding me!" Max and Luna were surprised but happy to have a new friend

**sample 2**: Once upon a time, there was a little boy named Tim. Tim loved to play with his toy cars. One day, he saw a big, red switch on the wall. He wondered what it could do.
Tim decided to flip the switch. When he did, a big, sour candy appeared! Tim was so happy. He had never seen a candy like this before. He decided to share the candy with his friends.
Tim shared the sour candy with his friends. They all took a bite and made funny faces. They laughed and had fun. Tim was happy that he listened to his mom's advice. And from that day on, Tim always shared his candy with his friends.

### moon (unseen-high-freq, corpus_hits=463)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim loved to play outside. One day, he saw a big, round moon in the sky. He wanted to touch the moon, but he was too small.
Tim had a friend named Sam. Sam was very good at climbing trees. Tim asked Sam to help him reach the moon. Sam said, "Okay, let's climb the moon together!" They climbed up the tree and got closer to the moon.
When they got to the moon, they saw that it was not a moon at all. It was a big, round ball! Tim and Sam laughed and played with the ball. They had a fun day under the big, round moon.

**sample 1**: One day, a little star wanted to travel to the moon. The star was very tired. It saw a big moon far away and said, "I want to travel to the moon."
The moon said, "Okay, let's travel together." They swam and played in the water. The star and the moon had a good time.
But then, the moon started to miss the water. The moon said, "Come back, star. I miss you." The star was sad, but it said, "I want to see the moon." The moon smiled and said, "You can come back anytime you want." The star felt happy and traveled back home, where it was not tired anymore.

**sample 2**: Once upon a time, there was a pretty princess. She lived in a big, expensive castle. The princess had a mommy and a daddy. She liked to play with her toys and laugh all day.
One day, the princess went outside to play. She saw a big, pretty star in the sky. The star was so bright! She wanted to touch it, but she knew she had to be careful.
The princess decided to reverse her shoes. She put her shoes back on and walked backwards. She was happy to touch the star. The pretty sky was so pretty. And she played all day long.

### school (unseen-high-freq, corpus_hits=1109)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim was a very honest boy. He always told the truth. One day, Tim went to school. He was very happy.
At school, Tim met a new friend named Sam. Sam was a little boy who also liked to play. They played with toys and had fun. But then, something unexpected happened.
A big, funny dog came into the school. The dog was very friendly. The dog wanted to play with Tim and Sam. They all played together and had a great time. The honest boy, the little boy, was happy to have a new friend.

**sample 1**: One day, a little boy named Tim went to school. He saw a big desk and wanted to sit there. But his friend, Sarah, was sitting on the desk too. She looked upset.
Tim walked up to Sarah and asked, "Why are you upset?" Sarah said, "I can't sit here because I am sad." Tim felt bad for Sarah and wanted to help her.
Tim had an idea. He told Sarah, "Let's play a game. We can put our toys in the desk and see if they are in the desk." They played the game and had lots of fun. Sarah was happy and Tim felt good.

**sample 2**: Once upon a time, there was a little boy named Tim. Tim loved to play outside. One day, it was so gloomy outside. Tim wanted to go to the park to play. He put on his coat and hat, and took his umbrella.
At the park, Tim saw his friend, Sally. Sally said, "Hi Tim! Let's play together!" They played on the swings, the slide, and the sandbox. They had so much fun. Then, they found a big box. They opened it and saw a big, red ball. They wanted to play with it, but the ball started to tremble.
Tim and Sally tried to play with the ball, but it was too big. The ball kept trembling more and

### king (unseen-high-freq, corpus_hits=625)

**sample 0**: Once upon a time, there was a king. The king had a big, shiny crown. He loved his crown very much. One day, the king wanted to wear his crown to a party. He was very happy.
At the party, the king saw a big, dangerous snake. The snake wanted to eat the king's crown. The king was scared. He did not want the snake to hurt his crown. So, he thought of a plan.
The king used his crown to scare the snake away. The snake was scared and ran away. The king was happy. He wore his crown and danced around the party. The king and his crown had a fun party together.

**sample 1**: One day, a king and his people were very hungry. They wanted to eat a big cake. They saw a beautiful crown on a table. The crown was shiny and pretty.
The king said, "Let's wear the crown!" The people were happy and put the crown on their heads. They thought the crown was very special. They played with the crown all day.
But then, a little girl came to the king. She said, "That crown is mine! Please give it back!" The king felt sad but knew the girl was right. So, he gave the crown back to the little girl. They all became friends and played together. The king learned that sharing is better than keeping things for himself.

**sample 2**: Once upon a time, there was a king who had a very big castle. The king loved his castle very much. He wanted to keep it safe, so he decided to bury it.
One day, a little girl came to visit the king. She saw the big castle and said, "Wow, what a big castle! I love it!" The king was very happy. He said, "Thank you for letting me keep your castle. It is my luxury. It is my home."
The little girl wanted to help the king. She said, "King, let's bury the castle together!" So, the king and the girl dug a hole in the ground. They put the castle in the ground and covered it with dirt

### baby (unseen-high-freq, corpus_hits=1155)

**sample 0**: Once upon a time, there was a little baby. The baby was very small and helpless. The baby lived in a big crib with his mommy and daddy. The crib was in a room with a big bed and a soft bed.
One day, the baby saw a toy on the floor. He wanted to play with it. He tried to grab the toy, but it was too high. The baby started to cry. His mommy and daddy heard him and came to help.
Mommy and daddy helped the baby get the toy. They gave it to the baby and said, "You can play with the toy now." The baby stopped crying and played with the toy. He was very happy and not helpless anymore.

**sample 1**: One day, a baby was playing with his toys. He was having a lot of fun. Then, a big cat came into the room. The cat saw the baby and wanted to play too. But the baby did not want to play with the cat.
The baby tried to ignore the cat, but the cat was persistent. The cat tried again and again. The baby did not like the cat. The baby started to cry. The cat did not like that.
The cat said, "I am sorry, baby. I will not ignore you now." The baby stopped crying and played with the cat. The baby and the cat became friends. The baby learned to play with the cat, and they had lots of fun. The baby

**sample 2**: Once upon a time, there was a pretty princess. She lived in a big, expensive castle. The princess had a mommy and a daddy. She liked to play with her toys and laugh all day.
One day, the pretty princess went outside to play. She saw a baby bird. The baby bird could not fly. The princess had an idea. She wanted to help the baby bird. She picked it up and carried it to her mommy.
Mommy was very happy. She hugged the princess and said, "Thank you!" The princess smiled and played with her baby bird. They all had fun together. And they all lived happily ever after.

### farm (unseen-high-freq, corpus_hits=1187)

**sample 0**: Once upon a time, there was a big farm. On the farm, there was a cow named Bessie. Bessie was a very useful cow. She could help the farmer with his work.
One day, Bessie saw a little boy named Tim. Tim was sad. Bessie asked, "Why are you sad, Tim?" Tim said, "I lost my toy. Can you help me find it?" Bessie wanted to help Tim.
Bessie and Tim looked for the toy. They looked in the grass. They looked near the big tree. They looked near the pond. Finally, they found the toy. Tim was happy. He said, "Thank you, Bessie!"
Bessie and Tim became good friends. They played together every day. And

**sample 1**: One day, a little boy named Tim went to the farm with his mom. They saw many animals like cows, pigs, and chickens. Tim was very happy to see all the animals. He wanted to take some corn home to eat.
Tim's mom said, "Do not throw corn at the animals, Tim. It is not good for the animals." But Tim did not listen. He was ignorant about the animals. He threw corn at the chickens, but he did not see a big rock in the way.
The chickens heard the noise and came to the farm. They took some corn from the rock and ran away. Tim felt sad because he did not see where the chickens went. He sat down on the soft grass and cried

**sample 2**: Once upon a time, there was a farmer named Tom. He had a big red barn. Tom was a very honest farmer. He always told the truth.
One day, Tom had to go to the farm. He needed to take care of his animals. The animals were happy to be home. Tom started to pack his things. He put his clothes and toys away. He knew something fun was going to happen soon.
After a few days, Tom went to the farm. He saw a big, red apple on the ground. It was not an apple or apple. Tom picked up the apple and took a big bite. It was the best apple he ever had. The animals were happy too, because they loved the honest farmer.

### beach (unseen-high-freq, corpus_hits=1212)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to play in the sand at the beach. One day, she found a big, pretty shell. She was very happy and wanted to show it to her friend, Tom.
Lily ran to Tom's house and said, "Look, Tom! I found a big shell!" Tom looked at the shell and said, "Wow, that's a big shell! Let's play with it!" They both played with the shell, taking turns to push it and watch it roll.
As they played, Lily's mom called them for a snack. They left the shell on the sand and went to eat. But when they got to the water, they saw that the shell

**sample 1**: One day, a little boy named Tim went to the beach with his mom. The beach was big and had many sand. Tim liked to build sand castles and play in the sand. His mom always had a reliable job in the town to make sure all the sand castles were in.
While playing, Tim saw a big wave coming. It was going to destroy the sand castle. Tim was scared, but he knew his mom was close by. He wanted to save his sand castle from the big wave.
Tim ran to the water and picked up his sand castle. He tried to make it strong again. His mom helped him too. Together, they built the biggest, most strong sand castle ever. Tim was happy he could fix the wave

**sample 2**: Once upon a time, there was a pretty beach. The pretty beach had lots of sand and water. The pretty beach was very happy.
One day, a little girl named Lily came to play. She saw the pretty beach and wanted to go on it. But her mom said, "You can't go on the pretty beach, Lily." Lily was sad, but she listened to her mom.
Later, Lily saw the pretty beach again. She was so happy! She ran to the beach and played all day. The pretty beach was a fun place to be. And Lily listened to her mom's advice.

### dragon (unseen-mid-freq, corpus_hits=431)

**sample 0**: Once upon a time, there was a big dragon. The dragon was very hungry. He wanted to find some food. He looked high and low, but he could not find any.
One day, the dragon saw a little girl. She was eating a yummy cake. The dragon wanted to eat the cake too. But the cake was too big for him. The dragon was sad.
Then, the dragon had an idea. He asked the little girl to help him. The little girl was nice. She helped the dragon eat the cake. The dragon was happy. They became friends and played together. The dragon was not hungry anymore.

**sample 1**: One day, a little dragon was feeling very nervous. He was all alone in a big cave. He did not know where to go. He was scared, but he knew he had to find a new home.
The dragon met a big dragon. The big dragon said, "Why are you nervous, little dragon?" The little dragon said, "I don't know where to go. Can I stay with you?" The big dragon said, "I will take you to a new cave."
They went to the new cave. It was big and dark. The little dragon was still nervous. They saw a light at the end of the cave. The little dragon was not nervous anymore. He was happy to have a new home with his

**sample 2**: Once upon a time, there was a pretty princess. She lived in a big castle. The princess was not pretty, so she was often embarrassed. One day, she met an old dragon. The dragon was not pretty like the other dragons. He was sad because he did not have a pretty crown.
The princess wanted to help the dragon. She had an idea. She would use her pretty dress to make the dragon look nice. So, the princess put on her pretty dress and went outside.
The princess wanted to help the dragon. She used her pretty dress to wrap around the dragon. When the dragon saw the pretty dress, he was very happy. He said, "Thank you, princess!" The princess felt good because she helped

### robot (unseen-mid-freq, corpus_hits=273)

**sample 0**: Once upon a time, there was a little boy named Tim. Tim had a big, red robot. He loved to play with his robot every day. One day, Tim saw a big, scary shadow on the wall. He was scared and wanted to hide.
Tim's mom saw him looking at the wall. She said, "Tim, don't be scared. I will help you." She took a deep breath and walked over to the wall. She said, "Don't be scared, Tim. I will help you."
Tim's mom stood by the wall and talked to the other kids. She said, "When you are scared, ask the robot to come and play." The other kids agreed and came to play with Tim

**sample 1**: One day, a little boy named Tim went to play outside. He saw a big red car. It was an ambulance. Tim wanted to see it up close. He started to complain to his mom. "Mom, I want to see the ambulance," he said.
His mom thought for a moment and said, "I have an idea, Tim. Let's go see the ambulance. Maybe we can see inside." They went closer and saw that the ambulance was making a loud sound. It was making the noise. Tim and his mom felt safe because they were not alone. They took a minute to listen to the happy sound.
Later, when they were inside, Tim's mom told him a secret. "Tim, I have a

**sample 2**: Once upon a time, there was a loyal dog named Buddy. Buddy loved to play with his best friend, a little girl named Lucy. They played together every day and were very happy.
One day, Lucy and Buddy went to the park. They saw a big slide and wanted to play on it. Lucy said, "Buddy, let's go up the slide and see what happens." Buddy barked happily and they climbed the steps to the top of the slide.
When they got down the slide, they saw a big, shiny box with a funny note. The note said, "You found the slide after all." Lucy and Buddy ran to the box and opened it. Inside the box was a new toy for them to play with!

### pirate (unseen-mid-freq, corpus_hits=177)

**sample 0**: Once upon a time, there was a pirate named Tom. Tom had a big boat. He liked to sail on the sea. One day, Tom saw a pirate. The pirate was sad. Tom wanted to help the sad pirate.
Tom said, "Why are you sad, pirate?" The pirate said, "I am sad because I lost my treasure. I cannot find it." Tom wanted to help the pirate. He said, "I will help you find your treasure."
Tom and the pirate sailed on the sea. They looked for the treasure. They saw a big fish. The fish said, "I know where the treasure is!" The fish took the pirate to a big treasure chest. The pirate was happy. He said,

**sample 1**: One day, a little boy named Tim went to the beach with his mom. The beach was big and had many sand. Tim liked to build sand castles and play in the water. His mom always told him to be careful not to get too close to the water.
While Tim was playing, he heard a strange sound. It was an unknown sound. He didn't know what it was. He wanted to find out, so he followed the sound.
As he got closer to the sound, he saw a big box on the sand. Tim opened the box and found a toy pirate inside. The pirate had a hat, a coat, and a hat. Tim was so happy to find the toy pirate. He thanked the unknown sound for

**sample 2**: Once upon a time, there was a pirate named Tom. He was a crazy pirate who loved to play tricks on his friends. One day, Tom was playing with his friend Sam when they heard a voice.
"Hello, pirates!" said the voice. Tom and Sam looked up and saw a big pirate standing there. "We are playing pirates and having fun!" said Tom. The pirate nodded his head.
"Let's play pirates together!" said Sam. They played pirate games and had a great time. But then, something unexpected happened. A big wave came and washed away their pirate friends! Tom and Sam were sad.
Tom had an idea. He asked the pirate to come back and play. The pirate agreed and went back

### train (unseen-mid-freq, corpus_hits=594)

**sample 0**: Once upon a time, there was a little train. The train loved to go fast on the track. One day, the train saw a big, dangerous hill. The train was scared, but it wanted to go up the hill.
The train started to go up the hill. It went faster and faster. The train was very scared now. But then, something unexpected happened. The train saw a big, friendly bear on the track. The bear was not dangerous.
The bear was nice and just wanted to play. The train and the bear became friends. They went up the hill together. The train was not scared anymore. They had fun and played all day.

**sample 1**: One day, a little train was going down a long track. The train was very tired. It saw a big station far away and said, "I want to stop and rest."
A nice man came to the station. He saw the tired train and said, "Do not worry, little train. I will help you stop." The man started to load the train with food and water. The train was happy to see the food and water.
The man and the train became friends. They sat and talked on the track. The train said, "Thank you for helping me, nice man." The man smiled and said, "You're welcome, little train. You are a good friend." They both knew that helping others can make

**sample 2**: Once upon a time, there was a little train. The train liked to travel slow and make a lot of noise. One day, the train met a big, happy train.
"Hi, train!" said the little train. "Do you like to travel slow and make noise?"
"Yes, I do!" said the train. "But I like to make noise too. It's fun to make noise, not make noise. It's fun to play with!"
The big, happy train thought about it. The next day, the train would make noise, but not too loud. The little train learned that it was fun to be loud, but it also learned that being kind was better.
The moral of the story is

### rainbow (unseen-mid-freq, corpus_hits=706)

**sample 0**: Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big rainbow in the sky. It was so pretty! Lily wanted to touch the rainbow.
Lily asked her friend, Tom, "Can you help me touch the rainbow?" Tom said, "Yes, let's try together!" They both tried to reach the rainbow, but it was too high. They jumped and jumped, but they could not reach it.
Then, Lily had an idea. She said, "Let's use a big stick to help us reach the rainbow!" They found a stick and tried to reach the rainbow. With the stick, they touched the rainbow and it was so pretty!

**sample 1**: One day, a little dog named Max went for a jog in the park. The sun was shining, and the birds were singing. Max loved to run and play with his friends. As Max jogged, he saw a big rainbow in the sky. He felt dizzy after running so much. Max stopped and looked at the rainbow. It was so pretty!
Max decided to follow the rainbow to see where it went. As he followed the rainbow, he saw a big tree with a door. Max was very surprised! Max found a small room with a kind lady. The lady saw Max and smiled. She said, "Hello, little dog! Do you like the rainbow?" Max nodded, wagging his tail. The lady opened the door and let

**sample 2**: Once upon a time, there was a pretty princess named Lily. She lived in a big castle with her mom and dad. Lily loved to play outside, but she was scared to go alone. One day, she decided to set out for herself.
As Lily walked, she met a new friend, a little bird. The bird said, "Hi, Princess Lily! Why are you scared to go to school?" Lily said, "I am scared to go." The bird smiled and said, "Don't worry, we can show you the way."
Lily and the bird walked together, and they saw many pretty things. They saw a big tree, a red ball, and a green tree. Lily was not scared anymore. She was

### zorp (unseen-oov, corpus_hits=0)

**sample 0**: Once upon a time, there was a little girl named Lily. She had a big, red balloon. Lily loved her balloon very much. She played with it every day.
One day, Lily's balloon got stuck in a tree. She was very sad. She tried to get it down, but she could not. Lily's friend, Tom, saw her and asked, "Why are you sad, Lily?"
Lily said, "My balloon is stuck in the tree. I can't get it down." Tom wanted to help Lily. He had an idea. He found a long stick and used it to poke the balloon. The balloon fell down, and Lily was very happy.
Lily and Tom played with the balloon all day.

**sample 1**: One day, a little dog named Max went for a jog in the park. The sun was shining, and the birds were singing. Max loved to run and play with his friends. As Max jogged, he saw a big, rough rock. He thought it was very special.
Max decided to bite the rough rock. He thought it would be fun. But when he bit the rock, something unexpected happened. The rock turned into a big, rough turtle! Max was surprised, but he was happy.
The turtle said, "Hello, Max! I was hiding as a rock, but now I can talk!" Max was very surprised. He didn't know that turtles can talk. The turtle and Max became friends and played together in the

**sample 2**: Once upon a time, there was a little boy named Tim. Tim loved to play outside. One day, it was so gloomy outside. Tim wanted to go to the park to play.
At the park, Tim saw a big tree. He heard a noise behind the tree. He was scared, but he wanted to see what was making the noise. Tim walked to the tree and saw a little bird.
"Hi, little bird," said Tim. "You are making the noise. Can I play?"
The little bird looked at Tim and said, "Sure, but only if you promise to be quiet."
Tim promised to be quiet. The little bird played quietly for a while. Then, they played together all day.
