<?php

class CompetenciasDocumentos extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'competenciasDocumentos';
	}

	public function rules()
	{
		return array(
			array('documento, competencia', 'required'),
			array('documento, competencia', 'numerical', 'integerOnly'=>true),
			array('id, documento, competencia', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'documento0' => array(self::BELONGS_TO, 'Documentos', 'documento'),
			'competencia0' => array(self::BELONGS_TO, 'Competencias', 'competencia'),
		);
	}

	public function behaviors()
	{
		return array('CAdvancedArBehavior',
				array('class' => 'ext.CAdvancedArBehavior')
				);
	}

	public function attributeLabels()
	{
		return array(
			'id' => Yii::t('app', 'ID'),
			'documento' => Yii::t('app', 'Documento'),
			'competencia' => Yii::t('app', 'Competencia'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('documento',$this->documento);

		$criteria->compare('competencia',$this->competencia);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
