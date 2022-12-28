from tree import DecisionTreeGenerator
from enum import Enum
import utype

raw_melon_data = """
青绿,蜷缩,浊响,清晰,凹陷,硬滑,好瓜 
乌黑,蜷缩,沉闷,清晰,凹陷,硬滑,好瓜 
乌黑,蜷缩,浊响,清晰,凹陷,硬滑,好瓜 
青绿,蜷缩,沉闷,清晰,凹陷,硬滑,好瓜 
浅白,蜷缩,浊响,清晰,凹陷,硬滑,好瓜 
青绿,稍蜷,浊响,清晰,稍凹,软粘,好瓜 
乌黑,稍蜷,浊响,稍糊,稍凹,软粘,好瓜 
乌黑,稍蜷,浊响,清晰,稍凹,硬滑,好瓜 
乌黑,稍蜷,沉闷,稍糊,稍凹,硬滑,坏瓜 
青绿,硬挺,清脆,清晰,平坦,软粘,坏瓜 
浅白,硬挺,清脆,模糊,平坦,硬滑,坏瓜 
浅白,蜷缩,浊响,模糊,平坦,软粘,坏瓜 
青绿,稍蜷,浊响,稍糊,凹陷,硬滑,坏瓜 
浅白,稍蜷,沉闷,稍糊,凹陷,硬滑,坏瓜 
乌黑,稍蜷,浊响,清晰,稍凹,软粘,坏瓜 
浅白,蜷缩,浊响,模糊,平坦,硬滑,坏瓜 
青绿,蜷缩,沉闷,稍糊,稍凹,硬滑,坏瓜
"""


class Color(Enum):
    green = '青绿'
    black = '乌黑'
    white = '浅白'


class Root(Enum):
    curl = '蜷缩'
    little_curl = '稍蜷'
    hard = '硬挺'


class Sound(Enum):
    muddy = '浊响'
    dull = '沉闷'
    crispy = '清脆'


class Texture(Enum):
    clear = '清晰'
    little_blur = '稍糊'
    blur = '模糊'


class Umbilicus(Enum):
    flat = '平坦'
    little_hollow = '稍凹'
    hollow = '凹陷'


class Tactile(Enum):
    hard = '硬滑'
    soft = '软粘'


class Quality(Enum):
    good = '好瓜'
    bad = '坏瓜'


class MelonModel(utype.Schema):
    # num: int  # (编号)
    color: Color      # 色泽
    root: Root       # 根蒂
    sound: Sound      # 敲声
    texture: Texture    # 纹理
    umbilicus: Umbilicus      # 脐部
    tactile: Tactile    # 触感

    quality: Quality    # 好瓜/坏瓜
    # good: bool


def test_watermelon(
    test_indexes: list = None,      # 用于测试集的索引
    pre_prune: bool = False,        # 是否预剪枝
    post_prune: bool = False,       # 是否后剪枝
):
    generator = DecisionTreeGenerator(
        model_cls=MelonModel,
        classifier='quality'
    )
    melon_values = []
    for line in raw_melon_data.splitlines():
        if not line:
            continue
        fields = line.strip().split(',')
        melon_values.append(tuple(fields))
    root = generator.generate(
        melon_values,
        test_indexes=test_indexes,
        pre_prune=pre_prune,
        post_prune=post_prune
    )
    print(root)
    return root


if __name__ == '__main__':
    test_watermelon([3, 4, 7, 8, 10, 11, 12], False, True)
