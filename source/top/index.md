---
title: 热度排行
comments: false
---

{% block content %}  {######################}  {### ABOUT BLOCK ###}  {######################}
<div id="top"></div>
<script src="http://cdn1.lncld.net/static/js/av-core-mini-0.6.4.js"></script>
<script>AV.initialize("A0NQ8NrwSuSevoacmYvPv0Qk-gzGzoHsz", "2b7GYcEUoHtDSGAJDIEH1TAB");
</script>
<script type="text/javascript">
var showTop = function(){
  var showCount=25;
  var time=0;
  var index = 0;
  var title="";
  var url="";
  var query = new AV.Query('Counter');
  query.notEqualTo('id',0);
  query.descending('time');
  query.limit(showCount);
  query.find().then(function (todo) {
      console.log(todo);
      console.log(todo);
      for (var i=0;i<showCount; i++){
        var result=todo[i].attributes;
        time=result.time;
        title=result.title;
        url=result.url;
        index++;
        var content="<p>"+"<font color='#1C1C1C'>"+"【文章热度:"+time+"℃】"+"</font>"+"<a href='"+"https://www.yynnyy.cn"+url+"'>"+title+"</a>"+"</p>";
        document.getElementById("top").innerHTML+=content;
      }
    }
    )
    .fail(function (object, error) {
      console.log("Error: " + error.code + " " + error.message);
    });
  }
setTimeout(showTop,1000);
</script>
{##########################}  {### END ABOUT BLOCK ###}  {##########################}{% endblock %}