import re
import os

def rewrite(txt_filename, html_filename):
    tmp_filename = 'tmp.txt'
    tmp2         = 'tail.txt'
    os.system('touch '+tmp2)
    with open(tmp_filename, 'a+') as tmp_f:
        with open(html_filename, 'r') as html_f:
            for line in html_f:
                if 'rewrite HERE' in line:
                    print('rewrite HERE')
                    os.system('tail -n1 '+txt_filename+' > '+tmp2)
                    with open(tmp2, 'r') as tmp2_f:
                        last = tmp2_f.readline()[:-1] # no newline
                    pattern = '<h2>[a-zA-Z\' ]+</h2>'
# the pattern needs to match all sorts of weird characters the user might type
# currently I only support letters, spaces, and the apostrophe '
                    print('line (orig) = ' + line)
                    new_line =\
                      re.sub(
                        pattern,                # prev
                        '<h2>'+str(last)+'</h2>',    # replacement
                        line)
                    print('new_line = ' + new_line)
                    tmp_f.write(new_line)
#                   tmp_f.write(
#                     re.sub(
#                       pattern,                # prev
#                       '<h2>'+str(last)+'</h2>',    # replacement
#                       line))
# if 'rewrite HERE' not in line:
                else:
                    tmp_f.write(line)
    os.system('cp -f '+tmp_filename+' '+html_filename)
    os.system('rm -f '+tmp_filename)
    os.system('rm -f '+tmp2)

if __name__=='__main__':
    txt_filename = "/home/n/txt.txt"#"/home/ubuntu/icml2016/scripts/cub_queries.txt"
    # TODO:  integrate txt_filename with embedding network and then the generator
    html_filename = '/home/n/templates/show.html'
    rewrite(txt_filename, html_filename)
