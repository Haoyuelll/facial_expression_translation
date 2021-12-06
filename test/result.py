import os

fileList = []
for root, dirs, files in os.walk("/home/hazel/hazel/Learning/CV/CUT/celeba_CUT_checkpoints/vgg_1/ori_B", topdown=False):
    print(files)
    epList = []
    for name in files:
        epList.append(name[:2])
        fileList.append(name)
    epList.sort(reverse=True)
    print(epList)
    fileList.sort(reverse=True)

with open('/home/hazel/hazel/Learning/CV/CUT/celeba_CUT_checkpoints/vgg_1/index.html', 'r') as html:
    hl = html.readlines()
    hList = []
    for id in epList:
        for i, line in enumerate(hl):
            if line[15:17] == str(id):
                print(line[15:17])
                hList.append(int(i))

    with open('/home/hazel/hazel/Learning/CV/CUT/celeba_CUT_checkpoints/vgg_1/result.html', 'w') as result:
        for i in range(6):
            result.write(hl[i])
        for h, name in zip(hList, fileList):
            pic = f'''        <td halign="center" style="word-wrap: break-word;" valign="top">
          <p>
            <a href="ori_B/{name}">
              <img src="ori_B/{name}" style="width:200px">
            </a><br>
            <p>ori_B</p>
          </p>
        </td>
      </tr>
    </table>
            '''
            for i in range(35):
                ind = hl[h+i].find('256px')
                if ind != -1:
                    hl[h+i]=hl[h+i][:ind]+'200px'+hl[h+i][ind+5:]
                result.write(hl[h+i])

            result.write(pic)
            
        result.write(hl[-2])
        result.write(hl[-1])
        result.close()
    html.close()

    #     print(os.path.join(root, name))