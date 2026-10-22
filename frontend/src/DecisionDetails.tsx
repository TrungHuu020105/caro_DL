import type { Analysis } from "./types";

export default function DecisionDetails({ analysis }: { analysis: Analysis }) {
  const number = (value: number | null | undefined) =>
    value == null ? "—" : value.toFixed(3);
  return (
    <details className="decision-details">
      <summary>Chi tiết điểm và tìm kiếm</summary>
      <p>
        Điểm CNN/chiến thuật được chuẩn hóa để so sánh ứng viên. Policy thể hiện
        ưu tiên chọn ô.
      </p>
      {analysis.weights && (
        <p>
          Trọng số CNN: {Math.round(analysis.weights.neural * 100)}% · Chiến
          thuật: {Math.round(analysis.weights.tactical * 100)}%
        </p>
      )}
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Ô</th>
              <th>Policy</th>
              <th>CNN</th>
              <th>Chiến thuật</th>
              <th>Kết hợp</th>
              <th>Tìm kiếm</th>
            </tr>
          </thead>
          <tbody>
            {analysis.candidates.map((c) => (
              <tr
                key={`${c.row}-${c.col}`}
                className={c.selected ? "selected-candidate" : ""}
              >
                <td>
                  {String.fromCharCode(65 + c.col)}
                  {c.row + 1}
                  {c.selected ? " ✓" : ""}
                </td>
                <td>{(c.probability * 100).toFixed(1)}%</td>
                <td>{number(c.neural_score)}</td>
                <td>{number(c.tactical_score)}</td>
                <td>{number(c.combined_score)}</td>
                <td>{number(c.search_score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <small>Kéo ngang bảng để xem đủ các cột điểm.</small>
      <p>
        Value: {number(analysis.value)} ·{" "}
        {analysis.value_used
          ? "Đã dùng value ở các thế cờ trong tìm kiếm"
          : "Nước này được chọn bởi policy hoặc chiến thuật trực tiếp"}
        .
      </p>
      {analysis.search && analysis.search.nodes > 0 && (
        <p>
          Hoàn thành độ sâu {analysis.search.completed_depth} ·{" "}
          {analysis.search.nodes} nút · Ngân sách {analysis.search.budget_ms}{" "}
          ms.
        </p>
      )}
      <small>
        Value là điểm kết quả kỳ vọng theo dữ liệu rollout, chưa phải xác suất
        thắng đã hiệu chỉnh.
      </small>
    </details>
  );
}
