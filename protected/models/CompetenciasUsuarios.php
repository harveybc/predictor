<?php

class CompetenciasUsuarios extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'competenciasUsuarios';
	}

	public function rules()
	{
		return array(
			array('usuario, competencia', 'required'),
			array('usuario, competencia', 'numerical', 'integerOnly'=>true),
			array('id, usuario, competencia', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'usuario0' => array(self::BELONGS_TO, 'Usuarios', 'usuario'),
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
			'usuario' => Yii::t('app', 'Usuario'),
			'competencia' => Yii::t('app', 'Competencia'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('usuario',$this->usuario);

		$criteria->compare('competencia',$this->competencia);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
