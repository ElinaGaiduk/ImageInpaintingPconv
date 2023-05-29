## Референс: https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/pconv_layer.py
from tensorflow.keras import backend as K #Функции для различных операций
from tensorflow.keras.layers import InputSpec #Спецификация входных данных
from tensorflow.keras.layers import Conv2D #Слои свертки


class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):        
        """Адаптировано из исходного слоя _Conv() of Keras"""
        
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
            
        self.input_dim = input_shape[0][channel_axis]
        
        # Ядро изображения
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Ядро маски
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        # Вычисление размера отступа (padding) для достижения нулевого заполнения (zero-padding)
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
        )

        # Window size - используется для нормализации
        self.window_size = self.kernel_size[0] * self.kernel_size[1]
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        '''
        

        Этот участок кода относится к применению операции свертки на входных данных с использованием метода conv2d в Keras. Он описывает последовательность действий, необходимых для создания маски и применения сверток.
        
        В данном случае маска создается с помощью операции свертки, где все веса установлены равными 1. То есть каждое значение в маске равно 1.
        Далее входные данные X умножаются на маску перед применением сверток. Это означает, что каждое значение входных данных будет умножено на соответствующее значение в маске.
        Умножение маски на входные данные позволяет усилить или подавить определенные признаки в данных в зависимости от значений в маске.

        После умножения маски на входные данные значения маски могут выходить за диапазон [0, 1]. Чтобы привести значения к диапазону от 0 до 1, применяется операция обрезания (clipping).
        Обрезание значений маски между 0 и 1 гарантирует, что веса маски останутся валидными и не перекроются с другими операциями или потерями в процессе обучения.
        ''' 

        # Необходимо предоставить и изображение (image), и маску (mask)
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

        # Отступ (padding) применяется явно таким образом, чтобы он стал частью маскированной частичной свертки (masked partial convolution)

        '''
        В данном случае отступ применяется явно, то есть добавление дополнительных пикселей вокруг входного изображения осуществляется специально для того, 
        чтобы эти пиксели стали частью маскированной частичной свертки. Это означает, что при применении маскированной частичной свертки будут учтены не только доступные пиксели изображения, 
        но и пиксели отступа, которые могут быть частично или полностью видимыми в маске.
        '''
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Применение операции свертки к маске
        mask_output = K.conv2d(
            masks, self.kernel_mask, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Применение операции свертки к изображению
        img_output = K.conv2d(
            (images*masks), self.kernel, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )        

        # Вычисление отношения маски на каждом пикселе в выходной маске
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Ограничение выходных значений от 0 до 1
        mask_output = K.clip(mask_output, 0, 1)

        # Удаление значений отношения, где есть пробелы или отверстия
        mask_ratio = mask_ratio * mask_output

        # Нормализация выходного изображения
        img_output = img_output * mask_ratio

        # Применение смещения (bias) только к изображению (если такое решение было принято)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)
        
        # Применение функции активации к изображению
        if self.activation is not None:
            img_output = self.activation(img_output)
            
        return [img_output, mask_output]
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]

## Референс: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/conv_utils.py#L85
def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """
    Функция определяет длину выхода свертки (convolution) на основе длины входа и других параметров. 
    Аргументы: 
    - input_length: целое число, представляющее длину входного сигнала (например, длину входной последовательности или ширину изображения).
    - filter_size: целое число, представляющее размер фильтра или ядра свертки.
    - padding: одно из значений "same", "valid", "full", определяющее тип отступа (padding).
       "same" означает, что выходной сигнал будет иметь ту же длину, что и входной сигнал (с использованием отступа для сохранения размеров).
       "valid" означает, что отступ не будет применяться, и выходной сигнал будет иметь меньшую длину, чем входной сигнал.
       "full" означает, что входной сигнал будет дополнен отступом так, чтобы выходной сигнал имел большую длину, чем входной сигнал.
    - stride: целое число, представляющее шаг или сдвиг фильтра при проходе по входному сигналу.
    - dilation: целое число, представляющее скорость дилатации (dilation rate). Определяет, как расположены пиксели фильтра в процессе свертки.
    Возврат:
    Длина выхода свертки (целое число).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = (filter_size - 1) * dilation + 1
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride
