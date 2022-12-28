import utype
from enum import Enum
from tree import DecisionTreeGenerator


class Age(Enum):
    young = '青年'
    mid = '中年'
    old = '老年'


class Credit(Enum):
    ordinary = '一般'
    good = '好'
    excellent = '非常好'


class CreditModel(utype.Schema):
    # num: int    # (编号)

    age: Age        # 年龄
    has_job: bool    # 是否有工作    True/False
    has_house: bool  # 是否有房子
    credit: Credit     # 信用

    grant: bool     # 是否贷款


credit_data = [
    ('青年', False, False, '一般', False),
    ('青年', False, False, '好', False),
    ('青年', True, False, '好', True),
    ('青年', True, True, '一般', True),
    ('中年', False, False, '一般', False),
    ('中年', False, False, '好', False),
    ('中年', True, True, '好', True),
    ('中年', False, True, '非常好', True),
    ('老年', False, True, '非常好', True),
    ('老年', False, True, '好', True),
    ('老年', True, False, '好', True),
    ('老年', True, False, '非常好', True),
    ('老年', False, False, '一般', False),
]


def test_credit(
    test_indexes: list = None,      # 用于测试集的索引
    pre_prune: bool = False,        # 是否预剪枝
    post_prune: bool = False,       # 是否后剪枝
):
    generator = DecisionTreeGenerator(
        model_cls=CreditModel,
        classifier='grant'
    )
    root = generator.generate(
        credit_data,
        test_indexes=test_indexes,
        pre_prune=pre_prune,
        post_prune=post_prune
    )
    print(root)
    return root


if __name__ == '__main__':
    test_credit()

# DecisionTreeNode(attname=has_job)
# -- DecisionTreeNode<has_job=False>(attname=has_house)
#   -- DecisionTreeNode<has_house=False>(result=False)
#   -- DecisionTreeNode<has_house=True>(result=True)
# -- DecisionTreeNode<has_job=True>(result=True)

# DecisionTreeNode(attname=has_house)
# -- DecisionTreeNode<has_house=False>(attname=has_job)
#   -- DecisionTreeNode<has_job=False>(result=False)
#   -- DecisionTreeNode<has_job=True>(result=True)
# -- DecisionTreeNode<has_house=True>(result=True)

    # 数据量不足，构建测试集效果不好
    # print('2. with test set, no prune ---------')
    # test_credit([1, 3, 5, 6, 9])
    # print('3. with test set, pre-prune=True ---------')
    # test_credit([1, 3, 5, 6, 9], pre_prune=True)
    # print('4. with test set, post-prune=True ---------')
    # test_credit([1, 3, 5, 6, 9], post_prune=True)
