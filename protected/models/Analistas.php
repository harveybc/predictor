<?php

class Analistas extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'analistas';
	}

	public function rules()
	{
		return array(
			array('modulo', 'numerical', 'integerOnly'=>true),
			array('Analista, Proceso', 'length', 'max'=>50),
			array('Pto_trabajo', 'length', 'max'=>8),
			array('id, Analista, Proceso, Pto_trabajo, modulo', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
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
			'Analista' => Yii::t('app', 'Analista'),
			'Proceso' => Yii::t('app', 'Proceso'),
			'Pto_trabajo' => Yii::t('app', 'Pto Trabajo'),
			'modulo' => Yii::t('app', 'Modulo'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('Analista',$this->Analista,true);

		$criteria->compare('Proceso',$this->Proceso,true);

		$criteria->compare('Pto_trabajo',$this->Pto_trabajo,true);

		$criteria->compare('modulo',$this->modulo);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
