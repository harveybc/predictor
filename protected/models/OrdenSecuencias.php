<?php

class OrdenSecuencias extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'ordenSecuencias';
	}

	public function rules()
	{
		return array(
			array('secuencia, posicion', 'numerical', 'integerOnly'=>true),
			array('id, secuencia, posicion', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'documentoses' => array(self::HAS_MANY, 'Documentos', 'ordenSecuencia'),
			'metaDocs' => array(self::HAS_MANY, 'MetaDocs', 'ordenSecuencia'),
			'secuencia0' => array(self::BELONGS_TO, 'Secuencias', 'secuencia'),
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
			'secuencia' => Yii::t('app', 'Secuencia'),
			'posicion' => Yii::t('app', 'Posicion'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('secuencia',$this->secuencia);

		$criteria->compare('posicion',$this->posicion);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
