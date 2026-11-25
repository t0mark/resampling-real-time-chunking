"""논문_양식.docx 위에 RS-RTC 논문 원고를 채워 논문_RS-RTC.docx 를 만든다."""
import copy
import re
import sys
from pathlib import Path

import docx
from docx.shared import Cm
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from lxml import etree

PAPER_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = PAPER_DIR / "논문_양식.docx"
OUTPUT_PATH = PAPER_DIR / (sys.argv[1] if len(sys.argv) > 1 else "논문_RS-RTC.docx")
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# ---------------------------------------------------------------- 원고 내용
TITLE_KO = "순차 몬테카를로 기반 역전파 없는 비동기 Action Chunking 유도"
TITLE_EN = "Gradient-Free Inpainting Guidance for Asynchronous Action Chunking via Sequential Monte Carlo"
ABSTRACT = (
    "Action chunking 정책을 멈춤 없이 실행하려면 현재 chunk를 실행하는 동안 다음 chunk를 "
    "생성해야 하며, 새 chunk는 그사이 실행된 행동과 이어져야 한다. 기존 방법은 이를 inpainting "
    "문제로 보고 불일치 비용의 기울기로 생성 과정을 유도하지만, 신경망 역전파가 필요하고 유도를 강하게 할수록 "
    "성능이 떨어진다. 본 논문은 일관된 chunk 생성을 불일치 비용으로 재가중한 정책 분포에서의 표본 추출로 "
    "정식화하고, 이를 역확산 단계에 걸친 순차 몬테카를로로 근사한다. 정책의 역확산 전이를 제안 분포로 사용하므"
    "로 순전파만으로 동작하며 표본이 학습된 분포를 벗어나지 않는다. 추론 지연을 모사한 시뮬레이션에서 제안 방법"
    "은 기울기 유도와 동등한 성공률을 보이면서 chunk 경계의 불일치와 불연속을 줄였다."
)
KEYWORDS = "Imitation learning, Diffusion models, Sequential Monte Carlo, Robot manipulation, Real-time control"

# 본문 블록: (종류, 내용). 본문 표기: *기울임*, _{아래첨자}, ^{위첨자}
BODY = [
    ("h1", "서론"),
    ("p", "Action chunking은 정책이 한 번의 추론으로 여러 시점의 행동열(chunk)을 생성하여 시간적으로 일관된 행동을 얻는 방법이다[1]. 추론에 시간이 걸리는 정책을 멈춤 "
          "없이 실행하려면 현재 chunk를 실행하는 동안 다음 chunk를 생성해야 하며, 이때 생성 중에 실행된 행동과 새 chunk가 어긋나면 chunk 경계에서 행동이 불연속이 된다."
          " RTC[2]는 이를 inpainting 문제로 보고, 이미 확정된 행동과의 불일치 비용의 기울기로 생성 과정을 유도(guidance)한다. 그러나 이 방식은 신경망 역전파를 "
          "요구하며, 유도를 강하게 할수록 표본이 학습된 행동 분포를 벗어난다."),
    ("p", "본 논문은 확정된 행동과 일관된 chunk 생성을 정책 분포를 불일치 비용으로 재가중한 목표 분포의 표본 추출로 정식화하고, 이를 순차 몬테카를로(SMC)로 근사하는 RS-RTC를"
          " 제안한다. 제안 방법은 정책의 역확산 전이를 제안 분포로 사용하므로 순전파만으로 동작하며, 표본이 정책 분포를 벗어나지 않는다."),
    ("funding", None),
    ("h1", "선행 연구"),
    ("p", "BID[3]는 후보 chunk를 독립적으로 생성한 뒤 이전 chunk와의 일관성을 기준으로 하나를 선택하며, 생성 과정 자체는 유도하지 않는다. RTC[2]는 각 역확산 단계의 "
          "예측 끝점과 확정된 행동 사이 가중 오차의 기울기로 표본을 보정한다. 확산 모델의 조건부 표본 추출을 SMC로 수행하는 방법[4]은 점근적으로 정확하지만 제안 분포를 구성할 때 "
          "기울기를 사용한다."),
    ("h1", "제안 방법"),
    ("h2", "3.1 문제 정의"),
    ("p", "정책 *p*_{θ}(*A*|*o*)는 관측 *o*에서 길이 *H*의 행동열 *A* = (*A*_{1}, …, *A*_{H})를 *K*단계 역확산(DDIM[5])으로 생성하며, "
          "*s* 스텝마다 다음 chunk 생성을 시작한다. 생성에 *d* 스텝이 걸리면 그동안 이전 chunk의 행동이 실행되므로 새 chunk의 앞 *d*개 행동은 이미 확정되어 있다. "
          "이전 chunk의 남은 행동을 목표 *Y*, 위치별 가중치를 *W*(확정 구간 1, 이후 겹치는 구간에서 감소, 그 외 0)[2]라 하면 불일치 비용은 식 (1)이다."),
    ("eq", "1"),
    ("p", "비동기 chunk 생성은 정책 분포를 유지하면서 불일치가 작은 행동열에 더 큰 확률을 부여하는 식 (2)의 목표 분포에서 표본을 얻는 문제가 된다. λ는 일치 강도이다."),
    ("eq", "2"),
    ("h2", "3.2 SMC 기반 유도"),
    ("p", "식 (2)는 정규화 상수를 알 수 없으므로 *N*개 입자로 역확산 단계 *k*마다 중간 목표 분포를 추적한다. 제안 분포로 정책의 확률적 역확산 전이(η > 0)를 그대로 쓰고 "
          "중간 potential을 식 (3)으로 두면, 정책 확률이 약분되어 가중치는 식 (4)와 같이 potential 증분만으로 갱신된다. 여기서 *x̂*_{0|k}는 단계 *k*의 "
          "예측 끝점이며, λ_{k}는 단계가 진행될수록 λ까지 증가한다."),
    ("eq", "3"),
    ("eq", "4"),
    ("p", "유효 표본 크기 1/Σ_{i}(*ŵ*^{(i)})^{2}가 *γN* 미만이면 재표본추출로 비용이 작은 입자를 복제하고, 역확산 전이의 확률성이 복제된 입자를 다시 분화시킨다. 최종"
          " 단계에서는 비용이 가장 작은 입자를 출력한다(그림 1). 가중치는 비용 값만으로 계산되므로 미분이 필요 없다."),
    ("fig", ("fig0_method_overview.png",
             "그림 1. (a) 유도 없음, (b) 기울기 유도, (c) 제안 방법.")),
    ("h1", "실험"),
    ("h2", "4.1 실험 설정"),
    ("p", "평면 밀기 과제(PushT)에서 사전학습된 확산 정책(*K* = 10, *H* = 16, *s* = 8)을 사용하였다. 추론 지연 *d* ∈ {0, 1, 2, 4, 7}을 모사하고"
          " 지연마다 500 에피소드를 평가하였다. 비교 방법은 유도 없는 비동기 실행, BID(*N* = 64), RTC(β = 5)이며, 제안 방법은 *N* = 64, λ = 1000, "
          "η = 0.5, *γ* = 0.5를 사용하였다(λ, β는 검증 에피소드에서 선택). 평가 지표는 성공률, 경계 불연속(경계 행동 변화량), 확정 구간 불일치(식 (1))이다."),
    ("h2", "4.2 실험 결과"),
    ("tab", None),
    ("p", "성공률은 모든 *d*에서 RTC와 제안 방법의 95% 신뢰구간이 겹쳐 유의한 차이가 없었다(표 1, 그림 2). 반면 경계 불연속과 확정 구간 불일치는 모든 *d*에서 제안 방법이"
          " 가장 작았으며, *d* = 7에서 RTC 대비 각각 6%, 11% 감소하였다. 기울기 유도는 β를 키우면 성공률이 떨어지는데(검증 시 β = 50에서 52%), 표본을 비용의 "
          "기울기 방향으로 직접 이동시켜 학습된 행동 분포를 벗어나게 하기 때문이다. 반면 제안 방법은 정책이 생성한 표본을 재가중하여 선택하므로 분포를 유지한 채 일치도를 높인다. "
          "chunk 생성 시간은 *N* = 64에서 약 110 ms로 RTC(약 287 ms)보다 짧았다."),
    ("fig", ("fig1_delay_sweep.png",
             "그림 2. 지연 d에 따른 성공률(음영: 95% 신뢰구간), 경계 불연속, 확정 구간 불일치.")),
    ("h1", "결론"),
    ("p", "본 논문은 비동기 action chunking에서 확정된 행동과 일관된 chunk 생성을 재가중된 정책 분포의 표본 추출로 정식화하고, 이를 기울기 없이 근사하는 SMC 기반 "
          "유도를 제안하였다. 제안 방법은 기울기 유도와 동등한 성공률을 유지하면서 경계 일관성을 개선하였다. 가중치가 비용 값만으로 결정되므로 충돌 회피 등 미분 불가능한 실행 제약으로도 "
          "확장할 수 있다."),
]

TABLE_CAPTION = "표 1. 지연 d에 따른 RTC와 제안 방법의 비교"
TABLE_ROWS = [
    ("0", "75.8", "73.4", "10.02", "9.43", "3.9", "3.1"),
    ("1", "74.6", "74.0", "12.77", "11.73", "4.1", "3.3"),
    ("2", "76.0", "78.2", "15.96", "14.37", "4.9", "3.9"),
    ("4", "61.8", "65.0", "21.72", "20.22", "7.8", "6.7"),
    ("7", "41.0", "37.2", "32.70", "30.70", "16.8", "14.9"),
]

REFERENCES = [
    "C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy "
    "learning via action diffusion,” *Proc. of Robotics: Science and Systems*, 2023.",
    "K. Black, M. Y. Galliker, and S. Levine, “Real-time execution of action chunking flow policies,” "
    "*arXiv preprint arXiv:2506.07339*, 2025.",
    "Y. Liu, J. I. Hamid, A. Xie, Y. Lee, M. Du, and C. Finn, “Bidirectional decoding: Improving action chunking "
    "via closed-loop resampling,” *arXiv preprint arXiv:2408.17355*, 2024.",
    "L. Wu, B. L. Trippe, C. A. Naesseth, D. M. Blei, and J. P. Cunningham, “Practical and asymptotically exact "
    "conditional sampling in diffusion models,” *Advances in Neural Information Processing Systems*, 2023.",
    "J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” *Proc. of the International "
    "Conference on Learning Representations*, 2021.",
]


# ---------------------------------------------------------------- 수식 (OMML)
def math_run(text, plain=False):
    style = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return (f'<m:r>{style}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
            f'<w:sz w:val="20"/></w:rPr><m:t xml:space="preserve">{text}</m:t></m:r>')


def math_sub(base, sub):
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def math_subsup(base, sub, sup):
    return f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"


def math_frac(numerator, denominator):
    return f"<m:f><m:num>{numerator}</m:num><m:den>{denominator}</m:den></m:f>"


def math_sum(sub, body):
    return (f'<m:nary><m:naryPr><m:chr m:val="∑"/><m:supHide m:val="1"/></m:naryPr>'
            f"<m:sub>{sub}</m:sub><m:sup/><m:e>{body}</m:e></m:nary>")


def math_hat(base):
    return f'<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>{base}</m:e></m:acc>'


def build_equation(number):
    r = math_run
    upper_i = r("(") + r("i") + r(")")
    if number == "1":
        squared_error = (r("‖", True) + math_sub(r("Y"), r("j")) + r("−", True) + math_sub(r("A"), r("j"))
                         + r("‖", True))
        body = (r("c") + r("(", True) + r("A") + r(")", True) + r("=", True)
                + math_frac(math_sum(r("j"), math_sub(r("W"), r("j"))
                                     + f"<m:sSup><m:e>{squared_error}</m:e><m:sup>{r('2', True)}</m:sup></m:sSup>"),
                            math_sum(r("j"), math_sub(r("W"), r("j")))))
    elif number == "2":
        body = (r("π") + r("(", True) + r("A") + r("|", True) + r("o") + r(")", True) + r("∝", True)
                + math_sub(r("p"), r("θ")) + r("(", True) + r("A") + r("|", True) + r("o") + r(")", True)
                + r("exp", True) + r("(", True) + r("−", True) + r("λ") + r("c") + r("(", True) + r("A")
                + r(")", True) + r(")", True))
    elif number == "3":
        body = (math_subsup(r("G"), r("k"), upper_i) + r("=−", True) + math_sub(r("λ"), r("k"))
                + r("c") + r("(", True) + math_subsup(math_hat(r("x")), r("0|") + r("k"), upper_i)
                + r(")", True) + r(",  ", True) + math_sub(r("λ"), r("k")) + r("=", True) + r("λ")
                + math_frac(r("k") + r("+1", True), r("K")))
    else:
        body = (math_subsup(r("w"), r("k"), upper_i) + r("∝", True)
                + math_subsup(r("w"), r("k") + r("−1", True), upper_i) + r("exp", True) + r("(", True)
                + math_subsup(r("G"), r("k"), upper_i) + r("−", True)
                + math_subsup(r("G"), r("k") + r("−1", True), upper_i) + r(")", True))
    return f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W_NS}">{body}</m:oMath>'


# ---------------------------------------------------------------- 문단 조작
def clear_runs(paragraph_element):
    for child in list(paragraph_element):
        if child.tag != qn("w:pPr"):
            paragraph_element.remove(child)


def make_run(base_rpr, text, italic=False, bold=False, vert_align=None, latin_font=None):
    run = etree.SubElement(etree.Element(qn("w:p")), qn("w:r"))
    rpr = copy.deepcopy(base_rpr) if base_rpr is not None else etree.Element(qn("w:rPr"))
    run.append(rpr)
    for tag in ("w:color", "w:u"):
        for element in rpr.findall(qn(tag)):
            rpr.remove(element)
    if italic:
        rpr.append(etree.Element(qn("w:i")))
    if bold:
        for element in rpr.findall(qn("w:b")):
            rpr.remove(element)
        rpr.insert(1, etree.Element(qn("w:b")))
    if vert_align:
        vert = etree.SubElement(rpr, qn("w:vertAlign"))
        vert.set(qn("w:val"), vert_align)
    if latin_font:
        fonts = rpr.find(qn("w:rFonts"))
        fonts.set(qn("w:ascii"), latin_font)
        fonts.set(qn("w:hAnsi"), latin_font)
    reorder_rpr(rpr)
    text_element = etree.SubElement(run, qn("w:t"))
    text_element.text = text
    text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return run


RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "strike", "noProof", "color", "spacing", "w",
             "kern", "position", "sz", "szCs", "highlight", "u", "vertAlign"]


def reorder_rpr(rpr):
    children = list(rpr)
    children.sort(key=lambda element: RPR_ORDER.index(etree.QName(element).localname)
                  if etree.QName(element).localname in RPR_ORDER else 99)
    for element in children:
        rpr.remove(element)
        rpr.append(element)


MARKUP_PATTERN = re.compile(r"\*([^*]+)\*|_\{([^}]*)\}|\^\{([^}]*)\}")


def append_marked_text(paragraph_element, base_rpr, text):
    position = 0
    for match in MARKUP_PATTERN.finditer(text):
        if match.start() > position:
            paragraph_element.append(make_run(base_rpr, text[position:match.start()]))
        italic_text, sub_text, sup_text = match.groups()
        if italic_text is not None:
            paragraph_element.append(make_run(base_rpr, italic_text, italic=True))
        elif sub_text is not None:
            paragraph_element.append(make_run(base_rpr, sub_text, vert_align="subscript"))
        else:
            paragraph_element.append(make_run(base_rpr, sup_text, vert_align="superscript"))
        position = match.end()
    if position < len(text):
        paragraph_element.append(make_run(base_rpr, text[position:]))


def first_run_rpr(paragraph_element):
    run = paragraph_element.find(qn("w:r"))
    return copy.deepcopy(run.find(qn("w:rPr"))) if run is not None else None


def new_paragraph(prototype, text=None, base_rpr=None):
    paragraph_element = copy.deepcopy(prototype)
    rpr = base_rpr if base_rpr is not None else first_run_rpr(prototype)
    clear_runs(paragraph_element)
    if text:
        append_marked_text(paragraph_element, rpr, text)
    return paragraph_element


def set_center(paragraph_element):
    ppr = paragraph_element.find(qn("w:pPr"))
    for element in ppr.findall(qn("w:jc")):
        ppr.remove(element)
    jc = etree.Element(qn("w:jc"))
    jc.set(qn("w:val"), "center")
    anchor = ppr.find(qn("w:rPr"))
    if anchor is not None:
        anchor.addprevious(jc)
    else:
        ppr.append(jc)


def keep_next(paragraph_element):
    ppr = paragraph_element.find(qn("w:pPr"))
    ppr.find(qn("w:pStyle")).addnext(etree.Element(qn("w:keepNext")))
    return paragraph_element


def set_indent_zero(paragraph_element):
    ind = paragraph_element.find(qn("w:pPr")).find(qn("w:ind"))
    for attribute in list(ind.attrib):
        del ind.attrib[attribute]
    ind.set(qn("w:left"), "0")
    ind.set(qn("w:right"), "0")
    ind.set(qn("w:firstLine"), "0")


# ---------------------------------------------------------------- 표
def build_table(base_rpr):
    column_widths = [430, 690, 690, 690, 690, 690, 690]
    border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    borders = "".join(border.format(side) for side in ("top", "left", "bottom", "right", "insideH", "insideV"))
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in column_widths)
    table = etree.fromstring(
        f'<w:tbl xmlns:w="{W_NS}"><w:tblPr><w:tblW w:w="{sum(column_widths)}" w:type="dxa"/><w:jc w:val="center"/>'
        f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
        f'<w:tblCellMar><w:left w:w="28" w:type="dxa"/><w:right w:w="28" w:type="dxa"/></w:tblCellMar>'
        f'<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid></w:tbl>')

    cell_rpr = copy.deepcopy(base_rpr)
    for element in cell_rpr.findall(qn("w:spacing")) + cell_rpr.findall(qn("w:sz")):
        cell_rpr.remove(element)
    size = etree.SubElement(cell_rpr, qn("w:sz"))
    size.set(qn("w:val"), "18")
    reorder_rpr(cell_rpr)

    def add_cell(row, text, width, span=1, vmerge=None, bold=False, italic=False):
        cell = etree.SubElement(row, qn("w:tc"))
        span_xml = f'<w:gridSpan w:val="{span}"/>' if span > 1 else ""
        merge_xml = (f'<w:vMerge w:val="{vmerge}"/>' if vmerge == "restart" else "<w:vMerge/>") if vmerge else ""
        cell.append(etree.fromstring(f'<w:tcPr xmlns:w="{W_NS}"><w:tcW w:w="{width}" w:type="dxa"/>{span_xml}'
                                     f'{merge_xml}<w:vAlign w:val="center"/></w:tcPr>'))
        paragraph_element = etree.fromstring(
            f'<w:p xmlns:w="{W_NS}"><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
            f'<w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="center"/></w:pPr></w:p>')
        if text:
            paragraph_element.append(make_run(cell_rpr, text, bold=bold, italic=italic))
        cell.append(paragraph_element)

    header_top = etree.SubElement(table, qn("w:tr"))
    add_cell(header_top, "d", column_widths[0], vmerge="restart", italic=True)
    for label in ("성공률 (%)", "경계 불연속 (px)", "확정 구간 불일치 (×10⁻³)"):
        add_cell(header_top, label, column_widths[1] * 2, span=2)
    header_bottom = etree.SubElement(table, qn("w:tr"))
    add_cell(header_bottom, "", column_widths[0], vmerge="continue")
    for _ in range(3):
        add_cell(header_bottom, "RTC", column_widths[1])
        add_cell(header_bottom, "RS-RTC", column_widths[1])
    for row_values in TABLE_ROWS:
        row = etree.SubElement(table, qn("w:tr"))
        add_cell(row, row_values[0], column_widths[0])
        for column_index, value in enumerate(row_values[1:]):
            is_rs_rtc_metric = column_index >= 2 and column_index % 2 == 1
            add_cell(row, value, column_widths[1], bold=is_rs_rtc_metric)
    return table


# ---------------------------------------------------------------- 문서 생성
def main():
    document = docx.Document(str(TEMPLATE_PATH))
    body = document.element.body
    children = list(body.iterchildren())

    # 제목·초록 영역
    title_ko, title_en, title_en_tail, abstract, keywords = (children[1], children[2], children[3],
                                                              children[7], children[9])
    title_rpr = first_run_rpr(title_ko)
    clear_runs(title_ko)
    title_ko.append(make_run(title_rpr, TITLE_KO))
    title_en_rpr = first_run_rpr(title_en)
    clear_runs(title_en)
    title_en.append(make_run(title_en_rpr, TITLE_EN))
    clear_runs(title_en_tail)
    for paragraph_element, text in ((abstract, ABSTRACT), (keywords, KEYWORDS)):
        runs = paragraph_element.findall(qn("w:r"))
        text_rpr = copy.deepcopy(runs[1].find(qn("w:rPr")))
        for run in runs[2:]:
            paragraph_element.remove(run)
        for element in paragraph_element.findall(qn("w:proofErr")):
            paragraph_element.remove(element)
        runs[1].find(qn("w:t")).text = "  "
        paragraph_element.append(make_run(text_rpr, text))

    # 본문 원형 문단
    heading1, heading2, body_text = children[12], children[33], children[14]
    figure_caption, table_caption, equation, empty_line = children[40], children[44], children[52], children[13]
    funding_frame, reference_heading, reference_item = children[31], children[89], children[91]
    final_section = children[98]
    body_rpr = first_run_rpr(body_text)

    # 기존 원고가 있으면 사용자가 입력한 저자·소속·사사 문구를 그대로 가져옴
    previous_path = PAPER_DIR / "논문_RS-RTC.docx"
    if previous_path.exists():
        previous_children = list(docx.Document(str(previous_path)).element.body.iterchildren())
        for index in (4, 5, 6):
            children[index].addprevious(copy.deepcopy(previous_children[index]))
            body.remove(children[index])
        previous_frames = [element for element in previous_children
                           if element.find(qn("w:pPr")) is not None
                           and element.find(qn("w:pPr")).find(qn("w:framePr")) is not None]
        if previous_frames:
            funding_frame = copy.deepcopy(previous_frames[0])
        print("[build_paper] 기존 원고의 저자·소속·사사 문구 유지")

    # 기존 본문 제거
    for element in children[12:98]:
        body.remove(element)

    new_elements = []
    after_table = False
    for block_type, content in BODY:
        if block_type == "h1":
            if new_elements:
                new_elements.append(copy.deepcopy(empty_line))
            new_elements.append(keep_next(new_paragraph(heading1, content)))
        elif block_type == "h2":
            new_elements.append(keep_next(new_paragraph(heading2, content)))
        elif block_type == "p":
            paragraph_element = new_paragraph(body_text, content, body_rpr)
            if after_table:
                paragraph_element.find(qn("w:pPr")).find(qn("w:spacing")).set(qn("w:before"), "160")
                after_table = False
            new_elements.append(paragraph_element)
        elif block_type == "funding":
            new_elements.append(funding_frame)
        elif block_type == "eq":
            paragraph_element = new_paragraph(equation)
            ppr = paragraph_element.find(qn("w:pPr"))
            for element in ppr.findall(qn("w:jc")):
                ppr.remove(element)
            spacing = ppr.find(qn("w:spacing"))
            spacing.set(qn("w:before"), "40")
            spacing.set(qn("w:after"), "40")
            for attribute in ("beforeLines", "afterLines"):
                spacing.attrib.pop(qn(f"w:{attribute}"), None)
            ppr.find(qn("w:pStyle")).addnext(etree.fromstring(
                f'<w:tabs xmlns:w="{W_NS}"><w:tab w:val="center" w:pos="2300"/>'
                f'<w:tab w:val="right" w:pos="4600"/></w:tabs>'))
            tab_run = make_run(body_rpr, "")
            tab_run.remove(tab_run.find(qn("w:t")))
            tab_run.append(etree.Element(qn("w:tab")))
            paragraph_element.append(tab_run)
            paragraph_element.append(etree.fromstring(build_equation(content)))
            paragraph_element.append(copy.deepcopy(tab_run))
            paragraph_element.append(make_run(body_rpr, f"({content})"))
            new_elements.append(paragraph_element)
        elif block_type == "fig":
            image_name, caption = content
            image_paragraph = new_paragraph(body_text)
            set_indent_zero(image_paragraph)
            set_center(image_paragraph)
            image_paragraph.find(qn("w:pPr")).find(qn("w:spacing")).set(qn("w:before"), "120")
            body.append(image_paragraph)
            Paragraph(image_paragraph, document._body).add_run().add_picture(
                str(PAPER_DIR / "figures" / image_name), width=Cm(8.1))
            body.remove(image_paragraph)
            caption_paragraph = new_paragraph(figure_caption, caption)
            set_center(caption_paragraph)
            caption_paragraph.find(qn("w:pPr")).find(qn("w:spacing")).set(qn("w:after"), "120")
            new_elements.extend([image_paragraph, caption_paragraph])
        elif block_type == "tab":
            caption_paragraph = keep_next(new_paragraph(table_caption, TABLE_CAPTION))
            set_center(caption_paragraph)
            new_elements.extend([caption_paragraph, build_table(body_rpr)])
            after_table = True

    # 참고문헌
    new_elements.append(copy.deepcopy(empty_line))
    new_elements.append(reference_heading)
    reference_rpr = first_run_rpr(reference_item)
    for index, reference in enumerate(REFERENCES, start=1):
        paragraph_element = new_paragraph(reference_item)
        paragraph_element.append(make_run(reference_rpr, f"[{index}]"))
        tab_run = make_run(reference_rpr, "")
        tab_run.remove(tab_run.find(qn("w:t")))
        tab_run.append(etree.Element(qn("w:tab")))
        paragraph_element.append(tab_run)
        append_marked_text(paragraph_element, reference_rpr, reference)
        new_elements.append(paragraph_element)

    for element in new_elements:
        final_section.addprevious(element)

    document.save(str(OUTPUT_PATH))
    print(f"[build_paper] 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
