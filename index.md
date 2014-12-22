---
layout: home
---

<div class="index-content blog">
    <div class="section">
        <ul class="artical-cate">
            <li class="on"><a href="/"><span>我的博客</span></a></li>
            <li style="text-align:center"><a href="/opinion"><span>观点</span></a></li>
            <li style="text-align:right"><a href="/project"><span>项目源码</span></a></li>
        </ul>

        <div class="cate-bar"><span id="cateBar"></span></div>

        <ul class="artical-list">
        {% for post in site.categories.blog %}
            <li>
                <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
                <div class="title-desc">{{ post.description }}</div>
            </li>
        {% endfor %}
        </ul>
    </div>
    <div class="aside">
    </div>
    <a herf="http://shartoo.github.io/photo/">查看我的拍照作品</a>
</div>
