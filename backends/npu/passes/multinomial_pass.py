import paddle
@paddle.incubate.passes.ir.RegisterPass
def generate_multinomial():
    def pattern(x):
        multinomial_op = paddle.incubate.passes.ir.PassDesc.OP.multinomial
        return multinomial_op(X=x)

    def replace(x):
        out = paddle.incubate.passes.ir.PassDesc.OP.multinomial_op(X=x)
        out.Attr("num_samples").MappedPattern(op="multinomial", name="num_samples")
        return out

    return pattern, replace
