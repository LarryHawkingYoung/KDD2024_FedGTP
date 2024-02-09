import copy, torch

def split(ary, indices_or_sections):
    import numpy.core.numeric as _nx
    Ntotal = len(ary)
    Nsections = int(indices_or_sections)
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = ([0] +
                        extras * [Neach_section+1] +
                        (Nsections-extras) * [Neach_section])
    div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

    sub_arys = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(ary[st:end])

    return sub_arys


def avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg