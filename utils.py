import torch
from torch.autograd import Variable
from tensorboard import summary

def add_scalar_summary(summary_writer, name, value, step):
    value = unwrap_scalar_variable(value)
    summ = summary.scalar(name=name, scalar=value)
    summary_writer.add_summary(summary=summ, global_step=step)

def add_histo_summary(summary_writer, name, value, step):
    value = value.view(-1).data.cpu().numpy()
    summ = summary.histogram(name=name, values=value)
    summary_writer.add_summary(summary=summ, global_step=step)

def wrap_with_variable(tensor, volatile, cuda):
    if isinstance(tensor, Variable):
        return tensor.cuda() if cuda else tensor
    elif cuda:
        return Variable(tensor.cuda(), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)

def wrap_with_variables(volatile, cuda, *tensors):
    return (wrap_with_variable(tensor, volatile, cuda) for tensor in tensors)


def unwrap_scalar_variable(var):
    if isinstance(var, Variable):
        return var.data[0]
    else:
        return var

def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length-1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    reversed_indices = Variable(reversed_indices)
    if inputs.is_cuda:
        device = inputs.get_device()
        reversed_indices = reversed_indices.cuda(device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs
