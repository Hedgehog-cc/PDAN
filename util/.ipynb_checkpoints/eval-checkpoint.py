import torch, os
import scipy.io

def image_classification_test(loader, feature_extractor,classifier):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            _,f = feature_extractor(inputs)
            outputs = classifier(f)
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy
