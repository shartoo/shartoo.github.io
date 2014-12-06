为何会不能添加评论？
>>>>查看 layout文件夹下的 post.html我们发现，文件最后一行有个 <script src="/js/post.js" type="text/javascript"></script>，但是
这个"/js/post.js"存放的是什么内容呢？打开后看到这一行：
//**评论的代码也删掉哦***
    window.disqus_shortname = 'shartoo'; // required: replace example with your forum shortname
原来这里的名字没修改。修改成自己的就可以了

