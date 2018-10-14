import copy

from chainer import cuda
from chainer.training import extensions
from chainer import variable

import numpy
import six


class ConfusionMatrix(extensions.Evaluator):

    default_name = 'confusion_matrix'

    def __call__(self, trainer=None):
        self.count = numpy.zeros((2, 2), dtype=numpy.int32)
        return super(ConfusionMatrix, self).__call__(trainer)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)

        it = copy.copy(iterator)

        for batch in it:
            in_arrays = self.converter(batch, self.device)
            if isinstance(in_arrays, tuple):
                in_vars = tuple(variable.Variable(x) for x in in_arrays)
                eval_func(*in_vars)
            elif isinstance(in_arrays, dict):
                in_vars = {key: variable.Variable(x)
                           for key, x in six.iteritems(in_arrays)}
                eval_func(**in_vars)
            else:
                in_var = variable.Variable(in_arrays)
                eval_func(in_var)

            #print target.y.data
            ys = numpy.argmax(cuda.to_cpu(target.y.data), axis=1)
            ts = cuda.to_cpu(in_vars[-1].data)
            for y, t in zip(ys, ts):
                self.count[y][t] += 1
        print(self.count)
        return {'tp': self.count[1][1], 'fp': self.count[1][0],
                'fn': self.count[0][1], 'tn': self.count[0][0]}

    def finalize(self):
        self.count = numpy.zeros((2, 2), dtype=numpy.int32)

if __name__ == '__main__':
    # read the log from stdin
    import json
    import sys

    print("\tD+\tD-")
    print("C+:\tTP\tFN")
    print("C-:\tFP\tTN")
    lobj = json.load(sys.stdin)
    for i, lent in enumerate(lobj):
        tp = int(lent['tp'])
        tn = int(lent['tn'])
        fp = int(lent['fp'])
        fn = int(lent['fn'])
        print("EPOCH {}-------------------------".format(i+1))
        print("C+:\t{}\t{}".format(tp, fn))
        print("C-:\t{}\t{}".format(fp, tn))
