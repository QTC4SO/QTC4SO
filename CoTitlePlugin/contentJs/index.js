console.log("hello")
console.log(document)
console.log(location)


function createPage (originalTitle,desc ,tag) {
    page = $('<div id="auto_div" style=""></div>')
    
    $('body').append(page)
	
     $.ajax({
            url: "http://127.0.0.1:5000/so",
            type: "post",
            data: {
                IncompleteTitle: originalTitle,
                Desc: desc,
                Tag: tag,
            },
            success: function (res) {
                console.log(res);
				AutoComplete("auto_div", "title", res.title_List);	
            },
            error: function (e) {
                console.log(e.responseText)
            }
        })
     
    //拖拽
    drag(auto_div)
}


        //智能补全
        function AutoComplete(auto, search, title_list) {
            var autoNode = $("#" + auto);   //缓存对象（弹出框）
            var n = 0;
            var old_value = $("#" + search).val();
            autoNode.empty();  //清空上次的记录
            for (var i in title_list) {
                var wordNode = title_list[i];   //弹出框里的每一条内容
                var newDivNode = $("<div>").attr("id", i);    //设置每个节点的id值
                newDivNode.attr("style", "font:14px/25px arial;height:25px;padding:0 8px;cursor: pointer;z-index:9999");
                newDivNode.html(wordNode).appendTo(autoNode);  //追加到弹出框
                //鼠标移入高亮，移开不高亮
                newDivNode.mouseover(function () {
                    $(this).css("background-color", "#ebebeb");
                });
                newDivNode.mouseout(function () {
                    $(this).css("background-color", "white");
                });
                //鼠标点击文字上屏
                newDivNode.click(function () {
                    autoNode.hide();
                    //取出高亮节点的文本内容
                    var comText = $(this).text();
                    //文本框中的内容变成高亮节点的内容
                    $("#" + search).val(comText);
                });
               //如果返回值有内容就显示出来
                autoNode.show();
            }
    }


//拖拽
function drag(ele) {
    let oldX, oldY, newX, newY
    ele.onmousedown = function (e) {
        if (!cj_move_page.style.right && !cj_move_page.style.bottom) {
            cj_move_page.style.right = 0
            cj_move_page.style.bottom = 0
        }
        oldX = e.clientX
        oldY = e.clientY
        document.onmousemove = function (e) {
            newX = e.clientX
            newY = e.clientY
            cj_move_page.style.right = parseInt(cj_move_page.style.right) - newX + oldX + 'px'
            cj_move_page.style.bottom = parseInt(cj_move_page.style.bottom) - newY + oldY + 'px'
            oldX = newX
            oldY = newY
        }
        document.onmouseup = function () {
            document.onmousemove = null
            document.onmouseup = null
        }
    }
}

$(document).keydown(function (e) {
        if (e.ctrlKey && e.keyCode === 81) {


            var incompletetitle = document.getElementById('title').value;
            var body1 = document.getElementsByClassName('s-code-block markdown')[0].innerText;
            var body2=document.getElementsByClassName('s-code-block markdown')[1].innerText;
            var tag=document.getElementById('tagnames').value;

            console.log(incompletetitle)
            console.log(body1+body2)
            console.log(tag)

            createPage(incompletetitle,body1+body2 ,tag)
        }
    })