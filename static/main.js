var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    if (!f.name.endsWith(".pcap")) { // 检查文件扩展名是否为“.pcap”
      window.alert("请选择后缀名为“.pcap”的文件。"); // 弹出提示框
      continue;
    }
    previewFile(f);
  }
}

var filePreview = document.getElementById("file-preview"); 
var fileDisplay = document.getElementById("file-display"); 
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");
var chartImg = document.getElementById("chartImg");
var currentDisplay = "no";

function submitFile() {
  console.log("submit");
  if (!fileDisplay.src) {
    window.alert("Please select an pcapfile before submit.");
    return;
  }
  loader.classList.remove("hidden");
  fileDisplay.classList.add("loading");
  var reader = new FileReader();
  predictFile(fileDisplay.src);
}

function clearFile() {
  fileSelect.value = "";
  filePreview.src = "";
  fileDisplay.src = "";
  predResult.innerHTML = "";
  hide(filePreview);
  hide(fileDisplay);
  hide(loader);
  hide(predResult);
  hide(chartImg);
  show(uploadCaption);
  fileDisplay.classList.remove("loading");
}


function previewFile(file) {
  console.log(file.name);

  var reader = new FileReader();
  reader.readAsArrayBuffer(file); // 读取为字节数组
  reader.onloadend = () => {
    var fileSize = (file.size / 1024).toFixed(2) + "KB";
    filePreview.innerHTML = "<div class='file-item'>" +
                            "<i class='iconfont icon-cz-tjhz'></i>" +
                            "<div class='file-name'>" + file.name + "</div>" +
                            "<div class='file-size'>" + fileSize + "</div>" +
                            "</div>";

    show(filePreview);
    hide(uploadCaption);

    // reset
    predResult.innerHTML = "";
    fileDisplay.classList.remove("loading");

    displayFileInfo( reader.result, file, "file-display");
  };
}

function predictFile(fileContent) {
  console.log(fileContent);
  fetch("/predict", {
    method: "POST",
    body: new Blob([fileContent]),
    headers: {
      "Content-Type": "application/octet-stream"
    }
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured：", err.message);
      window.alert("Oops! Something went wrong.");
    });
}


function displayFileInfo(src, file, id) {
  let display = document.getElementById(id);
  display.src = src
  display.querySelector("#file-list").innerHTML = "";
  var fileSize = (file.size / 1024).toFixed(2) + "KB";

  var fileIcon = "";
  if (file.type === "application/vnd.tcpdump.pcap") {
    fileIcon = "<i class='iconfont icon-cz-yccb'></i>";
  } else {
    fileIcon = "<i class='iconfont icon-wj-wj-1'></i>";
  }

  var fileItem = "<div class='file-item'>" +
                 fileIcon +
                 "<div class='file-name'>" + file.name + "</div>" +
                 "<div class='file-size'>" + fileSize + "</div>" +
                 "</div>";

  display.querySelector("#file-list").innerHTML += fileItem;
  show(display);
}

function switchDisplay() {
  if (currentDisplay === "result") { // 如果当前显示结果
    hide(predResult); // 隐藏结果
    show(chartImg); // 显示图像
    predResult.style.opacity = "0";
    currentDisplay = "chart"; // 更新当前显示元素
  } else if(currentDisplay === "chart") { // 如果当前显示图像
    hide(chartImg); // 隐藏图像
    show(predResult); // 显示结果
    predResult.style.opacity = "1";
    currentDisplay = "result"; // 更新当前显示元素
  }else {
    window.alert("You have to submit the file first.");
  }
}
function displayResult(data) {
  hide(loader);
  predResult.innerHTML = data.result;
  show(predResult);
  currentDisplay = "result"
  chartImg.src = "data:image/png;base64, " + data.graphic;
}

function hide(el) {
  el.classList.add("hidden");
}

function show(el) {
  el.classList.remove("hidden");
}
