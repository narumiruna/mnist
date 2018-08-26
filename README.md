# mnist

## Installation

```
# python setup.py install
```

## Usage

```python
from PIL import Image, ImageDraw

import mnist

dataset = mnist.MNIST('data', shape=(-1, 28, 28), one_hot=False)
data = dataset.train_images[0]
label = dataset.train_labels[0]

image = Image.fromarray(data)
draw = ImageDraw.Draw(image)
draw.text((0, 0), str(label), fill='white')
image.show()
```
