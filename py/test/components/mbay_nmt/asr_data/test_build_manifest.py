from pathlib import Path

from mbay_nmt.asr_data.build_manifest import ManifestRow, S3Object, _iter_mbay_rows, iter_xml_rows, join_s3


def test_mbay_combined_xml_distinguishes_audio_text_pairs(tmp_path: Path) -> None:
    xml = tmp_path / "Mbay.xml"
    xml.write_text(
        """<NewDataSet>
  <Entry>
    <ENTRY>àbÆlày</ENTRY>
    <ENTRYCODE>3342</ENTRYCODE>
    <SOUNDFILE>WordsA_18.wav</SOUNDFILE>
  </Entry>
  <Samples>
    <SAMPLESENT>«-já m“¾ kÆ «-jö ndògæ tÉ.</SAMPLESENT>
    <TRANS_SENT>I cut grass.</TRANS_SENT>
    <SOUNDFILE>MbayJ_593.wav</SOUNDFILE>
  </Samples>
  <Expressions>
    <IDIOM_EXP>jà gàjÆ/b¼y</IDIOM_EXP>
    <TRANS_EXP>circumcise</TRANS_EXP>
    <SAMPLESENT>Kº½-gæ à jà-ñ b¼y à•.</SAMPLESENT>
    <TRANS_SENT>The Kan don't perform circumcision.</TRANS_SENT>
    <SOUNDFILE>MbayJ_629.wav</SOUNDFILE>
    <SOUNDEXPR>MbayJ_627.wav</SOUNDEXPR>
  </Expressions>
</NewDataSet>""",
        encoding="utf-8",
    )

    rows = list(_iter_mbay_rows(xml))

    assert [(row.kind, row.soundfile_ref) for row in rows] == [
        ("headword", "WordsA_18.wav"),
        ("sentence", "MbayJ_593.wav"),
        ("expression", "MbayJ_627.wav"),
        ("sentence", "MbayJ_629.wav"),
    ]
    assert rows[0].text == "àbə̀lày"
    assert rows[2].text == "jà gàjə̀/bà̰y"


def test_iter_xml_rows_extracts_sister_language_rows(tmp_path: Path) -> None:
    (tmp_path / "Mbay.xml").write_text("<NewDataSet />", encoding="utf-8")
    (tmp_path / "NgambaySamp.xml").write_text(
        """<NewDataSet>
  <Table>
    <SAMPLESENT>ä-« jíngä «-tá m-är-ï là.</SAMPLESENT>
    <SOUNDFILE>Nga14SS01.wav</SOUNDFILE>
  </Table>
</NewDataSet>""",
        encoding="utf-8",
    )
    (tmp_path / "SarExpr.xml").write_text(
        """<NewDataSet>
  <Table>
    <IDIOM_EXP>kÆ ÷í</IDIOM_EXP>
    <SOUNDEXPR>SarExpr01.wav</SOUNDEXPR>
    <SAMPLESENT>Ngön g‡ bàsá.</SAMPLESENT>
    <SOUNDFILE>SarSent01.wav</SOUNDFILE>
  </Table>
</NewDataSet>""",
        encoding="utf-8",
    )

    rows = list(iter_xml_rows(tmp_path))

    assert [(row.language, row.kind, row.soundfile_ref) for row in rows] == [
        ("ngambay", "sentence", "Nga14SS01.wav"),
        ("sar", "expression", "SarExpr01.wav"),
        ("sar", "sentence", "SarSent01.wav"),
    ]


def test_join_s3_resolves_wav_refs_to_mp3_keys_case_insensitively() -> None:
    rows = list(
        join_s3(
            [
                ManifestRow(language="mbay", text="à", kind="headword", soundfile_ref="MBAY01A11.wav"),
                ManifestRow(language="mbay", text="ā", kind="headword", soundfile_ref="missing.wav"),
            ],
            {"sounds/mbay01a11.mp3": S3Object(key="sounds/mbay01a11.mp3", size=1234)},
            "sounds/",
        )
    )

    assert rows[0].s3_key == "sounds/mbay01a11.mp3"
    assert rows[0].duration_hint_bytes == 1234
    assert rows[1].s3_key is None
    assert rows[1].duration_hint_bytes is None
