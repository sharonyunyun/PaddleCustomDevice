import paddle
@paddle.incubate.passes.ir.RegisterPass
def generate_softmax():
    def pattern(x):
        softmax_op = paddle.incubate.passes.ir.PassDesc.OP.softmax
        return softmax_op(X=x)

    def replace(x):
        out = paddle.incubate.passes.ir.PassDesc.OP.softmax_op(X=x)
        out.Attr("axes").MappedPattern(op="softmax", name="axis")
        return out

    return pattern, replace
