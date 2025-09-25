// HTML要素の取得
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const resultsDiv = document.getElementById('results');
const chatModal = document.getElementById('chatModal');
const closeChatModalButton = document.getElementById('closeChatModal');
const chatTitle = document.getElementById('chatTitle');
const chatHistory = document.getElementById('chatHistory');
const chatInput = document.getElementById('chatInput');
const chatButton = document.getElementById('chatButton');


let currentPaperId = null;

// 検索ボタンがクリックされた時の処理
searchButton.addEventListener('click', async () => {
    const query = searchInput.value;
    if (!query) return; // 入力が空の場合は何もしない

    // 検索中であることをユーザーに通知
    resultsDiv.innerHTML = '<p class="text-center text-gray-500">検索中...</p>';

    try {
        // バックエンドAPIにPOSTリクエストを送信
        const response = await fetch('http://127.0.0.1:8000/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        // サーバーからの応答をJSONとして取得
        const data = await response.json();

        // 検索結果を画面に表示
        displayResults(data.results);

    } catch (error) {
        console.error('検索エラー:', error);
        resultsDiv.innerHTML = '<p class="text-center text-red-500">検索中にエラーが発生しました。サーバーが起動しているか確認してください。</p>';
    }
});

// 検索結果を表示する関数
function displayResults(papers) {
    resultsDiv.innerHTML = ''; // 前回の結果をクリア

    if (papers.length === 0) {
        resultsDiv.innerHTML = '<p class="text-center text-gray-500">検索結果がありませんでした。</p>';
        return;
    }

    papers.forEach(paper => {
        const paperElement = document.createElement('div');
        paperElement.className = 'bg-white p-4 rounded-lg shadow-md hover:shadow-xl transition-shadow cursor-pointer';
        paperElement.innerHTML = `
            <h3 class="text-xl font-bold text-blue-700">${paper.title}</h3>
            <p class="text-gray-600">${paper.filename}</p>
            <p class="text-gray-500 mt-2 line-clamp-3">${paper.fulltext || '要約がありません。'}</p>
        `;
        resultsDiv.appendChild(paperElement);
    });
}

// チャットモーダルを開く関数
function openChatModal(paper) {
    currentPaperId = paper.id;
    chatTitle.textContent = paper.title;
    chatHistory.innerHTML = '';
    chatModal.classList.remove('hidden');
    chatModal.classList.add('flex');
}

// チャットモーダルを閉じる関数
closeChatModalButton.addEventListener('click', () => {
    chatModal.classList.add('hidden');
    chatModal.classList.remove('flex');
});

// チャットボタンがクリックされた時の処理
chatButton.addEventListener('click', async () => {
    const question = chatInput.value;
    if (!question || !currentPaperId) return;

    // ユーザーのメッセージを会話履歴に追加
    chatHistory.innerHTML += `<div class="bg-blue-100 p-2 rounded-md mb-2 self-start">${question}</div>`;
    chatInput.value = '';

    // ローディング表示
    const loadingMessage = document.createElement('div');
    loadingMessage.className = 'text-gray-500 italic mb-2';
    loadingMessage.textContent = 'AIが回答を生成中...';
    chatHistory.appendChild(loadingMessage);

    try {
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paper_id: currentPaperId, question: question })
        });
        const data = await response.json();

        // ローディング表示を削除
        chatHistory.removeChild(loadingMessage);

        // AIの回答を会話履歴に追加
        chatHistory.innerHTML += `<div class="bg-gray-200 p-2 rounded-md mb-2">${data.answer}</div>`;
        chatHistory.scrollTop = chatHistory.scrollHeight; // スクロールを一番下に

    } catch (error) {
        console.error('Error during chat:', error);
        chatHistory.innerHTML += `<div class="bg-red-200 p-2 rounded-md mb-2">エラーが発生しました。もう一度お試しください。</div>`;
    }
});