with open('./loss_log.txt', 'r') as log:
    key_list = ['D_A', 'D_B', 'G_A', 'G_B',
                'cycle_A', 'cycle_B', 'idt_A', 'idt_B']
    lines = log.readlines()
    new_log = '{'
    i = 0
    for line in lines:
        if (line.find('=') != -1):
            new_log = '{'
            i = 0
            continue
        value_num = line.find('D_A')
        new_value = line[1:value_num-28] + line[value_num:-2]
        line = f'\"{i}\":{{{new_value}'
        line = line.replace('epoch', '\"epoch\"')
        line = line.replace('iters', '\"iters\"')
        for key in key_list:
            line = line.replace(key, ', \"'+key+'\"')
        line += '},'
        new_log += line
        i += 1
    new_log += '}'

with open('loss_log.json', 'w') as loss_log:
    loss_log.write(new_log)
    loss_log.close()
