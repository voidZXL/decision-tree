from typing import List, Type, Dict, Any, Union, Optional, Tuple
import math
from enum import Enum
from functools import cached_property


class DecisionTreeNode:
    def __init__(
        self,
        # attname: str,
        fields: List[str],
        classifier: str,
        values: List[dict],
        parent: 'DecisionTreeNode' = None,
        parent_value=None
    ):
        self.values = values
        self.parent = parent
        self.fields = fields
        self.classifier = classifier
        self.parent_value = parent_value
        self.children: List['DecisionTreeNode'] = []
        self.attname = None
        self.result = None

    @property
    def length(self):
        return len(self.values)

    @cached_property
    def root_entropy(self):
        # 根节点的信息熵
        cls_dict = {}
        for item in self.values:
            cls = item[self.classifier]
            cls_dict.setdefault(cls, 0)
            cls_dict[cls] += 1

        entropy = 0
        for cls, count in cls_dict.items():
            div = count / self.length
            entropy += div * math.log2(div)

        return -entropy

    def get_attr_gain(self, name: str):
        # 计算属性的信息增益
        val_dict: Dict[Any, Dict[Any, int]] = {}
        for item in self.values:
            value = item[name]
            cls = item[self.classifier]
            val_dict.setdefault(value, {}).setdefault(cls, 0)
            val_dict[value][cls] += 1

        cond_entropy = 0
        # 计算条件信息熵

        for value, cls_dict in val_dict.items():
            entropy = 0
            total = sum(cls_dict.values())
            for cls, count in cls_dict.items():
                div = count / total
                entropy += div * math.log2(div)
            cond_entropy += -entropy * (total / self.length)

        return self.root_entropy - cond_entropy

    def find_max_gain_attrs(self) -> List[str]:
        # 遍历属性找到信息增益最大的属性
        revert_gain_dict = {}
        for attr in set(self.fields).difference({self.classifier}):
            gain = self.get_attr_gain(attr)
            revert_gain_dict.setdefault(gain, []).append(attr)
        if not revert_gain_dict:
            return []
        max_gain = max(revert_gain_dict.keys())
        attrs = revert_gain_dict[max_gain]
        return attrs

    def print(self, depth: int = 0):
        if depth:
            prefix = "  " * (depth - 1) + '-- '
        else:
            prefix = ''
        ret = prefix + repr(self) + "\n"
        for child in self.children:
            ret += child.print(depth + 1)
        return ret

    def __str__(self):
        return self.print()

    @classmethod
    def _repr_value(cls, val):
        if isinstance(val, Enum):
            val = val.value
        return repr(val)

    def __repr__(self):
        if self.parent:
            cond = f'<{self.parent.attname}={self._repr_value(self.parent_value)}>'
        else:
            cond = ''
        attrs = {}
        if self.attname:
            attrs.update(attname=self.attname)
        if self.result is not None:
            attrs.update(result=self._repr_value(self.result))
        attr_repr = ', '.join([f'{key}={val}' for key, val in attrs.items()]) if attrs else ''
        return f'{self.__class__.__name__}{cond}({attr_repr})'

    def get_root(self) -> 'DecisionTreeNode':
        if self.parent:
            return self.parent.get_root()
        return self

    def generate(self, test_set: List[dict] = None,
                 pre_prune: bool = False,
                 post_prune: bool = False,
                 ):
        # 1. 检测是否已完成分类
        classes = set()
        for item in self.values:
            cls = item[self.classifier]
            classes.add(cls)
        if not classes:
            return
        if len(classes) == 1:
            self.result = classes.pop()
            return

        max_gain_attrs = self.find_max_gain_attrs()
        if not max_gain_attrs:
            return

        if len(max_gain_attrs) == 1:
            return self.generate_by_attr(
                max_gain_attrs[0],
                test_set=test_set,
                pre_prune=pre_prune,
                post_prune=post_prune
            )

        if not test_set:
            # 无法在没有测试集的情况下对相同信息增益属性作判断
            return self.generate_by_attr(max_gain_attrs[0])

        prediction_rate_map = {}
        attr_children_map = {}
        for attr in max_gain_attrs:
            self.generate_by_attr(
                attr,
                test_set=test_set,
                pre_prune=pre_prune,
                post_prune=post_prune
            )
            prediction_rate_map[attr] = self.test_prediction(test_set)
            attr_children_map[attr] = self.children

        max_rate = max(prediction_rate_map.values())
        chosen_attr = None
        for attr, rate in prediction_rate_map.items():
            if rate >= max_rate:
                chosen_attr = attr
        if chosen_attr:
            self.attname = chosen_attr
            self.children = attr_children_map[chosen_attr]
            print(f'choose the attribute: {repr(chosen_attr)} with rate: {max_rate}')

    def generate_by_attr(
        self,
        attname: str,
        test_set: List[dict] = None,
        pre_prune: bool = False,
        post_prune: bool = False,
    ):
        self.attname = attname
        value_set = {}
        for item in self.values:
            value = item[self.attname]
            value_set.setdefault(value, []).append(item)

        children = []
        for val, items in value_set.items():
            child = self.__class__(
                values=items,
                fields=list(set(self.fields).difference({self.attname})),
                classifier=self.classifier,
                parent=self,
                parent_value=val
            )
            children.append(child)

        if test_set and pre_prune:
            # 预剪枝
            self.children = []
            pre_correct_rate = self.test_prediction(test_set)  # 未形成子树时的预测正确率
            self.children = children
            post_correct_rate = self.test_prediction(test_set)  # 形成子树后的预测正确率
            if post_correct_rate <= pre_correct_rate:
                # 划分子树不能带来增益，剪枝
                print(f'pre-pruning: '
                      f'attname={self.attname}, '
                      f'pre-rate={pre_correct_rate}, '
                      f'post-rate={post_correct_rate}')
                self.children = []
        else:
            self.children = children

        for child in self.children:
            child.generate(test_set, pre_prune=pre_prune, post_prune=post_prune)  # recursively

        if self.children and test_set and post_prune:
            # 后剪枝
            pre_correct_rate = self.test_prediction(test_set)  # 当前子树的预测正确率
            memo = self.children
            self.children = []
            post_correct_rate = self.test_prediction(test_set)  # 剪枝后的预测正确率
            if post_correct_rate > pre_correct_rate:
                # 剪枝后带来增益，剪枝
                print(f'post-pruning: '
                      f'attname={self.attname}, '
                      f'pre-rate={pre_correct_rate}, '
                      f'post-rate={post_correct_rate}')
            else:
                self.children = memo

    def test_prediction(self, test_set: List[dict]):
        root = self.get_root()
        correct = 0
        for item in test_set:
            prediction, conf_rate = root.predict(**item)
            if prediction is not None and prediction == item.get(self.classifier):
                correct += conf_rate
        return correct / len(test_set)

    def predict(self, **kwargs) -> Tuple[Any, float]:     # 结果，置信率
        # 根据决策树进行预测
        if self.result is not None:
            return self.result, 1
        if self.attname and self.attname in kwargs:
            val = kwargs[self.attname]
            for child in self.children:
                if child.parent_value == val:
                    return child.predict(**kwargs)
        # depend on counts
        cls_dict = {}
        for item in self.values:
            cls = item[self.classifier]
            cls_dict.setdefault(cls, 0)
            cls_dict[cls] += 1
        max_cls = max(cls_dict.values())
        for cls, count in cls_dict.items():
            if count >= max_cls:
                return cls, max_cls / len(self.values)
        return None, 1


class DecisionTreeGenerator:
    import utype
    def __init__(self, model_cls: Type[utype.Schema], classifier: str):
        self.model_cls = model_cls
        self.fields = list(model_cls.__parser__.fields)
        self.classifier = str(classifier)

    @utype.parse
    def generate(self,
                 data: List[Union[tuple, dict]],
                 test_indexes: Optional[List[int]] = None,
                 pre_prune: bool = False,
                 post_prune: bool = False,
                 ) -> DecisionTreeNode:
        # 返回决策树的根节点

        values = []
        test_set = []
        for item in data:
            if isinstance(item, dict):
                values.append(self.model_cls(**item))
            else:
                values.append(self.model_cls(**{field: item[i] for i, field in enumerate(self.fields)}))

        if test_indexes:
            # divide test set based on the classifier
            train_set = []
            test_set = []
            for i, item in enumerate(values):
                if i in test_indexes:
                    test_set.append(item)
                else:
                    train_set.append(item)
            values = train_set
        node = DecisionTreeNode(
            fields=self.fields,
            classifier=self.classifier,
            values=values
        )
        node.generate(test_set, pre_prune=pre_prune, post_prune=post_prune)
        if test_set:
            print('prediction correct rate:', node.test_prediction(test_set))
        return node
