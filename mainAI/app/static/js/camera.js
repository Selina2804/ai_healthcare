navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  const video = document.getElementById("video");
  video.srcObject = stream;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  video.addEventListener("loadeddata", () => {
    setInterval(() => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const data = canvas.toDataURL("image/jpeg", 0.8); // 0.8 = chất lượng ảnh cao hơn

      fetch("/process_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: data }),
      })
        .then((res) => res.json())
        .then((data) => {
          const emotionText = document.getElementById("emotion");
          if (data.emotion === "Không nhìn thấy khuôn mặt") {
            emotionText.textContent = "🚫 Không nhìn thấy khuôn mặt!";
            emotionText.style.color = "red";
          } else {
            emotionText.textContent = `😊 Cảm xúc: ${data.emotion}`;
            emotionText.style.color = "green";
          }
        })

        .catch((err) => {
          console.error("❌ Lỗi khi gửi ảnh:", err);
        });
    }, 3000); // tăng thời gian giữa các frame để tránh overload
  });
});

function sendMessage() {
  const msg = document.getElementById("chatInput").value;
  fetch("/chat", {
    method: "POST",
    body: JSON.stringify({ message: msg }),
    headers: { "Content-Type": "application/json" },
  })
    .then((res) => res.json())
    .then((data) => {
      document.getElementById("reply").textContent = data.reply;
    });
}
