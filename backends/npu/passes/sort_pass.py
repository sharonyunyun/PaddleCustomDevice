import paddle
@paddle.incubate.passes.ir.RegisterPass
def generate_sort():
    def pattern(x):
        sort_op = paddle.incubate.passes.ir.PassDesc.OP.top_k_v2
        return sort_op(X=x).Output("Out")

    def replace(x):
        out = paddle.incubate.passes.ir.PassDesc.OP.sort_op(X=x)
        out.Attr("num").MappedPattern(op="top_k_v2", name="k")
        return out

    return pattern, replace
