
```python
import re
import jieba.posseg as pseg
import hanlp
class QueryFilter:
    def __init__(self):
        pass
    def judge_sentence_integrity(self, sentence):
        HanLP = hanlp.pipeline() \
            .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
            .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
            .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos')
        res = HanLP(sentence)
        pos = res['pos']
        verb = ['VA', 'VC', 'VE', 'VV']
        noun = ['NR', 'NT', 'NN']
        sum_pos = []
        for pos_mem in pos:
            sum_pos.extend(pos_mem)
        if not set(sum_pos) & set(verb) or not set(sum_pos) & set(noun):
            return 0
        else:
            return 1
    def pos_tag_text(self, text):
        sentences = text.split('。')
        contains_noun_or_verb = False
        pos_tagged_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            words = pseg.cut(sentence)
            pos_tags = []
            for word, flag in words:
                pos_tags.append((word, flag))
                # 检查是否有名词或动词
                if flag.startswith('n') or flag.startswith('v'):
                    contains_noun_or_verb = True
            pos_tagged_sentences.append(pos_tags)
        # 如果没有名词或动词，返回 0
        if not contains_noun_or_verb:
            return 0
        return 1
```
