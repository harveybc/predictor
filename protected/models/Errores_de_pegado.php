<?php

class Errores_de_pegado extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'errores_de_pegado';
	}

	public function rules()
	{
		return array(
			array('Campo0, Campo1, Campo2, Campo3, Campo4, Campo5, Campo6, Campo7, Campo8, Campo9, Campo10, Campo11, Campo12, Campo13, Campo14', 'safe'),
			array('id, Campo0, Campo1, Campo2, Campo3, Campo4, Campo5, Campo6, Campo7, Campo8, Campo9, Campo10, Campo11, Campo12, Campo13, Campo14', 'safe', 'on'=>'search'),
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
			'Campo0' => Yii::t('app', 'Campo0'),
			'Campo1' => Yii::t('app', 'Campo1'),
			'Campo2' => Yii::t('app', 'Campo2'),
			'Campo3' => Yii::t('app', 'Campo3'),
			'Campo4' => Yii::t('app', 'Campo4'),
			'Campo5' => Yii::t('app', 'Campo5'),
			'Campo6' => Yii::t('app', 'Campo6'),
			'Campo7' => Yii::t('app', 'Campo7'),
			'Campo8' => Yii::t('app', 'Campo8'),
			'Campo9' => Yii::t('app', 'Campo9'),
			'Campo10' => Yii::t('app', 'Campo10'),
			'Campo11' => Yii::t('app', 'Campo11'),
			'Campo12' => Yii::t('app', 'Campo12'),
			'Campo13' => Yii::t('app', 'Campo13'),
			'Campo14' => Yii::t('app', 'Campo14'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('Campo0',$this->Campo0,true);

		$criteria->compare('Campo1',$this->Campo1,true);

		$criteria->compare('Campo2',$this->Campo2,true);

		$criteria->compare('Campo3',$this->Campo3,true);

		$criteria->compare('Campo4',$this->Campo4,true);

		$criteria->compare('Campo5',$this->Campo5,true);

		$criteria->compare('Campo6',$this->Campo6,true);

		$criteria->compare('Campo7',$this->Campo7,true);

		$criteria->compare('Campo8',$this->Campo8,true);

		$criteria->compare('Campo9',$this->Campo9,true);

		$criteria->compare('Campo10',$this->Campo10,true);

		$criteria->compare('Campo11',$this->Campo11,true);

		$criteria->compare('Campo12',$this->Campo12,true);

		$criteria->compare('Campo13',$this->Campo13,true);

		$criteria->compare('Campo14',$this->Campo14,true);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
