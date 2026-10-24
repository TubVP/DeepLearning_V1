import argparse

def generate_code(problem_text):
    # Giả lập mô hình — sau này bạn sẽ thay bằng LLM thật
    if "sum of the two largest" in problem_text:
        return """def sum_of_two_largest(nums):
    return sum(sorted(nums)[-2:])"""
    else:
        return "# Model chưa học bài toán này"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.problem, "r", encoding="utf-8") as f:
        problem_text = f.read()

    code = generate_code(problem_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(code)

if __name__ == "__main__":
    main()
