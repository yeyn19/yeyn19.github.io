<script type="text/javascript">
    // Wait for the page to load first
    var _prevOnload = window.onload;
    window.onload = function() {
        var switchLang = document.getElementById("menu").getElementsByClassName("menu-item-switchlanguage")[0].childNodes[1];
        var href = window.location.href;
        var indexOfEn = href.toLowerCase().indexOf('/en/');
        if(indexOfEn !== -1) {

            // switchLang.href = href.replace('/en/', '/');
            switchLang.href = href.replace("/en/", "/tags/English");
        }
        else {
            // switchLang.href = href.replace(/(^http[s]?:\/\/[a-z0-9.]*[:?0-9]*\/)(.*)/i, '$1en/$2');
            switchLang.href = href.replace("/en/", "/tags/English");
        }
        switchLang.href = "/tags/English";
    }
</script>
