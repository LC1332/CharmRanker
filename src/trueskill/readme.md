我这边想建立一个输入有向图边表

即可输出每个点hidden score的函数

我初步有个可参考的方案在calculate_true_skill.py

这个方案可能依赖于边表输入顺序
所以你可以帮我改成把边表shuffle 5次分别计算后把每个latent的节点平均

我后面要做模拟实验，确保你的输入是最简的
我想的是每个node用一个字符串标记就可以 然后再输入一个 list of tuple的有向图的边表
然后输出一个 nodename -> score的dict就比较简单

并且编写一个测试确保你的函数是正确的

建立一个和函数同名的文件方便其他的程序来import

- 限制你的代码都在 src/trueskill文件夹