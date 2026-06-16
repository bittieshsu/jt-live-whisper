/* jt-live-whisper — GitHub Pages 共用腳本：語言切換 / 點圖放大 / 指令複製 */
(function(){
  // 中英雙語切換（記住偏好）
  var btn = document.getElementById('langBtn');
  var KEY = 'jtlw-doc-lang';
  if (localStorage.getItem(KEY) === 'en') document.body.classList.add('lang-en');
  function setBtn(){ if (btn) btn.textContent = document.body.classList.contains('lang-en') ? '繁體中文' : 'English'; }
  setBtn();
  if (btn) btn.addEventListener('click', function(){
    document.body.classList.toggle('lang-en');
    localStorage.setItem(KEY, document.body.classList.contains('lang-en') ? 'en' : 'zh');
    setBtn();
  });

  // 點擊擷圖放大（lightbox）
  var lb = document.getElementById('lightbox');
  if (lb){
    var lbImg = document.getElementById('lightbox-img');
    document.querySelectorAll('.shot').forEach(function(img){
      img.addEventListener('click', function(){
        lbImg.src = img.getAttribute('src');
        lbImg.alt = img.getAttribute('alt') || '';
        lb.classList.add('on');
      });
    });
    var close = function(){ lb.classList.remove('on'); lbImg.src=''; };
    lb.addEventListener('click', close);
    document.addEventListener('keydown', function(e){ if (e.key === 'Escape') close(); });
  }

  // 指令區塊：複製按鈕（去掉註解只複製指令）
  var en = function(){ return document.body.classList.contains('lang-en'); };
  document.querySelectorAll('pre').forEach(function(pre){
    var b = document.createElement('button');
    b.type = 'button'; b.className = 'copy-btn';
    var label = function(){ b.textContent = en() ? 'Copy' : '複製'; };
    label();
    b.addEventListener('click', function(){
      var clone = pre.cloneNode(true);
      clone.querySelectorAll('.c, .copy-btn').forEach(function(n){ n.remove(); });
      var cmd = clone.textContent.replace(/\n{2,}/g, '\n').trim();
      navigator.clipboard.writeText(cmd).then(function(){
        b.textContent = en() ? 'Copied' : '已複製'; b.classList.add('ok');
        setTimeout(function(){ label(); b.classList.remove('ok'); }, 1500);
      });
    });
    pre.appendChild(b);
  });
})();
