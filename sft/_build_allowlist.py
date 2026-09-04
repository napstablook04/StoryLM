"""生成扩建版主题白名单 data/topic_allowlist_expanded.json。

来源：_candidates.txt（4056 个候选词，已通过 df>=30 / 停用词 / 大写占比<0.4 过滤，
即语料中真实存在、非人名、非虚词）。人工通读 rank 1-2600（df>=106）逐词审定。

审定标准（沿用初版 163 词白名单，用户确认过）：
  收：具体名词（实物/动物/地点/食物/身体/人物职业）
      活动词（play/catch/hide/draw...）
      情感与品格词（scared/grumpy/loyal...）及可作主题的状态词（cold/dark/hungry）
  弃：叙事胶水词（said/went/seemed/happening...）、空泛形容词（big/nice/pretty）、
      颜色词、数字、时间副词、所有格（dog's）、对话标签（yelled/gasped）

硬约束：评测探针词（moon/school/king/baby/farm/beach/dragon/robot/pirate/train/
rainbow/zorp 及其变形）绝不进入白名单——它们是"白名单外主题"测试集。
控制词 bird/cake/park/sharing 保留在白名单内。
"""

import json
import re
from pathlib import Path

from prepare_sft_data import TOPIC_ALLOWLIST as ORIGINAL_163

PROBES = {
    "moon", "school", "king", "baby", "farm", "beach",
    "dragon", "robot", "pirate", "train", "rainbow", "zorp",
}
PROBE_STEMS = {  # 探针词的屈折变形同样禁入
    "moons", "schools", "kings", "babies", "farms", "beaches",
    "dragons", "robots", "pirates", "trains", "rainbows",
}

ADDITIONS = """
# 人物 / 家庭 / 职业
woman mother father sister twins lady child kid son parents owner neighbor
teacher doctor nurse farmer driver pilot sailor painter dancer fireman soldier
police clown person children grandma grandpa aunt uncle family

# 动物 / 生物
lion cow monkey horse elephant fox duck turtle owl crab ant bee wolf shark
whale deer giraffe pig sheep goat goose swan seal dove pigeon parrot penguin
panda zebra tiger crocodile alligator dolphin jellyfish lizard spider rat mice
mole mule puppy kitten dinosaur monster witch ghost knight princess prince
queen hero superhero fairy pets creature creatures

# 地点 / 场所
shop market mall restaurant library museum theater airport station office
hospital castle palace kingdom cabin hut tent bedroom bathroom backyard
playground sandbox stadium gym pool street road sidewalk path trail bridge
village city neighborhood zoo jungle ocean shore river lake island mountain
valley cliff creek meadow stream field parade festival picnic

# 食物 / 饮食
cookie cookies bread butter cheese eggs milk juice tea cocoa soup salad pizza
pie candy lollipop gum chocolate strawberry berries berry banana bananas apple
apples orange grapes lemon tomato tomatoes carrots carrot corn pepper salt
sugar honey jam sandwich sandwiches spaghetti popcorn cereal rice avocado
toast snack snacks treats dessert breakfast lunch meal fruit fruits vegetables
nuts peanut hay cream ice

# 物品 / 玩具 / 交通
truck boat ship plane jet rocket bicycle scooter wagon cart kite balloon
balloons bubbles drum trumpet radio camera phone telephone computer video
movie mailbox package envelope newspaper news card cards notebook pencil
eraser chalk paint brush comb scissors glue tape string thread yarn button
buttons zip lock key ring jar bottle cup plate bowl spoon fork pot pan oven
stove fridge refrigerator freezer soap towel umbrella jacket coat pants skirt
shirt socks boots gloves belt cap scarf sunglasses mask costume suit crown
jewel diamond gold silver treasure map sword shield arrow flashlight lamp
candle candles mirror pillow blanket rug sofa couch shelf drawer closet
cupboard ladder bench fence gate roof stairs porch garage attic bathtub
shower sink faucet hose bucket mop shovel rake hammer tools rope basket tray
bin cage tank leash collar bell whistle blocks puzzle puzzles stickers clay
dough seeds seed plant plants vine sunflower oak twig twigs petals branch
wheel wheels swing swings slide slides gift gifts present helmet haircut
snowman snowballs bus motorcycle ambulance apron vase jug tray napkin counter
menu chores

# 自然 / 天气
star stars cloud clouds storm thunder lightning sunset sunrise shadow shadows
fire flame smoke snow winter summer spring breeze wave waves sand dirt mud
dust soil weather

# 身体
arm arms leg legs feet ear ears eye eyes teeth fur paw paws wing wings
feather feathers beak belly stomach tummy knee neck shoulder cheek tongue
hair finger fingers toes

# 活动词（含常见动名词形式）
catch explore touch fix hide visit buy pretend draw drawing ride fight teach
drink cook cooking wash build dig sail bounce hop clap cheer celebrate
decorate exercise jog march bake taste travel practice shopping rescue save
solve escape protect respect forgive collect imagine giggle tickle wiggle
fetch sneak polish smell count write crying

# 情感 / 品格 / 状态
upset nervous confused grateful sick sleepy grumpy cozy calm afraid jealous
guilty relieved embarrassed frustrated bored cheerful stubborn lazy clumsy
glad scary mad quiet dark cold hungry thirsty ashamed gloomy miserable jolly
lively peaceful playful sneaky nosy bossy rude polite honest loyal obedient
impatient envious selfish careless helpless restless anxious thankful humble
compassionate generous thoughtful determined persistent shocked startled
overjoyed thrilled amazed fearful crazy dizzy naughty strange delighted

# 抽象（儿童故事的经典主题）
friendship kindness courage hope dream dreams mystery secret promise trust
truth lesson goal victory winner prize reward imagination happiness joy love
luck excitement laughter

# 其他高频故事元素
goodbye cry smart money coin coins penny surprise surprises match trick
tricks joke jokes stories music birthday holiday bedtime playtime spell wand
crystal
"""


def main():
    # 候选词集：已在语料侧通过 df/停用词/大写占比三道过滤的词
    candidates = set()
    for line in Path("_candidates.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            candidates.add(parts[1])

    # 解析 ADDITIONS（去掉 # 注释行）
    added, seen = [], set()
    for line in ADDITIONS.strip().splitlines():
        line = line.split("#", 1)[0]
        for word in re.findall(r"[a-z]+", line):
            if word not in seen:
                seen.add(word)
                added.append(word)

    # 校验 1：新增词必须在候选集中（防拼写错误 / 防混入未验证词）
    missing = [w for w in added if w not in candidates]

    # 校验 2：探针词绝不入表
    final = set(ORIGINAL_163) | (set(added) - set(missing))
    leaked = final & (PROBES | PROBE_STEMS)

    print(f"original 163 present: {len(ORIGINAL_163 & final)}/163")
    print(f"additions parsed: {len(added)}, rejected (not in candidates): {len(missing)}")
    if missing:
        print("  dropped:", " ".join(sorted(missing)))
    if leaked:
        raise SystemExit(f"FATAL: probe words leaked into allowlist: {leaked}")

    out = {
        "words": sorted(final),
        "size": len(final),
        "source": "manual curation of _candidates.txt rank 1-2600 (df>=106) + original 163",
        "excluded_probes": sorted(PROBES),
    }
    Path("data/topic_allowlist_expanded.json").write_text(
        json.dumps(out, indent=0), encoding="utf-8")
    print(f"saved data/topic_allowlist_expanded.json: {len(final)} words "
          f"({len(final) - 163} new vs original)")


if __name__ == "__main__":
    main()
