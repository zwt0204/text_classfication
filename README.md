# text_classfication文本分类
包含二分类、多分类以及多标签分类
单标签与多标签的区别在于损失函数的计算，以及全连接层的输出：
损失函数：
多标签：tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
单标签：tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
全连接层：
多标签：self.probabilities = tf.nn.sigmoid(self.scores)
                self.predictions = tf.round(self.probabilities, name="predictions")
单标签：self.pro = tf.nn.softmax(self.scores)
                self.predicitions = tf.argmax(self.pro, 1, name='predictions')
## TextRNN and TextRNN+attention
- data:{"text":"data", "id":1}

## TextCNN 数据并行
- data:{"text":"data", "id":1}

## TextRCNN
- data:{"text":"data", "id":1}

## XGBOOST
- data:{"text":"data", "id":1}

## SVM
- data:{"text":"data", "id":1}

## TransformerXL
- data:{"text":"data", "id":1}

## BERT_TextCNN
- data: https://github.com/cjymz886/text_bert_cnn

## BERT_TextRNN
- data:下载链接:[https://pan.baidu.com/s/11AuC5g47rnsancf6nfKdiQ] 密码:1vdg