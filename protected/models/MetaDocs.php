<?php

class MetaDocs extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}
        
	public function tableName()
	{
		return 'metaDocs';
	}

	public function rules()
	{
		return array(
			array('UT_Area,UT_Proceso,cerveceria,titulo,medio, idioma,documento, fechaRecepcion', 'required'),
                        array('documento,modulo, columna, fila,tipoContenido, fabricante, cerveceria, medio, idioma, disponibles, existencias, modulo, columna, fila, documento, usuario, revisado, userRevisado, eliminado, secuencia, ordenSecuencia,ubicacionT, ', 'numerical', 'integerOnly'=>true),
                        array('ISBN, EAN13', 'length', 'max'=>32),
			array('numPedido, numComision, version', 'length', 'max'=>64),
			array('UT_Equipo,UT_Tablero_TAG,UT_Motor_TAG,descripcion, titulo, autores', 'length', 'max'=>256),
			array('id, tipoContenido, fabricante, cerveceria, numPedido, numComision, ubicacionT, descripcion, titulo, version, medio, idioma, disponibles, existencias, modulo, columna, fila, documento, ruta, fechaCreacion, fechaRecepcion, autores, usuario, revisado, userRevisado, fechaRevisado, fechaCreacion, eliminado, secuencia, ordenSecuencia, ISBN, EAN13', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
                        'ruta0' => array(self::BELONGS_TO, 'Archivos', 'ruta'),
			'tipoContenido0' => array(self::BELONGS_TO, 'TipoContenidos', 'tipoContenido'),
			'fabricante0' => array(self::BELONGS_TO, 'Fabricantes', 'fabricante'),
			'cerveceria0' => array(self::BELONGS_TO, 'Cervecerias', 'cerveceria'),
			'ubicacionT0' => array(self::BELONGS_TO, 'UbicacionTec', 'ubicacionT'),
			'medio0' => array(self::BELONGS_TO, 'Medios', 'medio'),
			'idioma0' => array(self::BELONGS_TO, 'Idiomas', 'idioma'),
			'documento0' => array(self::BELONGS_TO, 'Documentos', 'documento'),
			'usuario0' => array(self::BELONGS_TO, 'Usuarios', 'usuario'),
			'userRevisado0' => array(self::BELONGS_TO, 'Usuarios', 'userRevisado'),
			'secuencia0' => array(self::BELONGS_TO, 'Secuencias', 'secuencia'),
			'ordenSecuencia0' => array(self::BELONGS_TO, 'OrdenSecuencias', 'ordenSecuencia'),
			'prestamoses' => array(self::HAS_MANY, 'Prestamos', 'metaDoc'),
			'tablasDeContenidos' => array(self::HAS_MANY, 'TablasDeContenido', 'metaDoc'),
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
			'tipoContenido' => Yii::t('app', 'Tipo Contenido'),
			'fabricante' => Yii::t('app', 'Fabricante'),
			'cerveceria' => Yii::t('app', 'Cerveceria'),
			'numPedido' => Yii::t('app', 'Num Pedido'),
			'numComision' => Yii::t('app', 'Num Comision'),
			'ubicacionT' => Yii::t('app', 'Ubicación Técnica'),
			'descripcion' => Yii::t('app', 'Descripcion de Documento'),
			'titulo' => Yii::t('app', 'Título'),
			'version' => Yii::t('app', 'Versión'),
			'medio' => Yii::t('app', 'Medio'),
			'idioma' => Yii::t('app', 'Idioma'),
			'disponibles' => Yii::t('app', 'Disponibles'),
			'existencias' => Yii::t('app', 'Existencias'),
			'modulo' => Yii::t('app', 'Módulo'),
			'columna' => Yii::t('app', 'Columna'),
			'fila' => Yii::t('app', 'Fila'),
			'documento' => Yii::t('app', 'Documento'),
			'ruta' => Yii::t('app', 'Nombre del archivo'),
			'fechaCreacion' => Yii::t('app', 'Fecha Creación'),
			'fechaRecepcion' => Yii::t('app', 'Fecha Recepción'),
			'autores' => Yii::t('app', 'Autor'),
			'usuario' => Yii::t('app', 'Usuario'),
			'revisado' => Yii::t('app', 'Revisado'),
			'userRevisado' => Yii::t('app', 'User Revisado'),
			'fechaRevisado' => Yii::t('app', 'Fecha Revisado'),
			'eliminado' => Yii::t('app', 'Eliminado'),
			'secuencia' => Yii::t('app', 'Secuencia'),
			'ordenSecuencia' => Yii::t('app', 'Orden Secuencia'),
                        'ISBN' => Yii::t('app', 'ISBN'),
			'EAN13' => Yii::t('app', 'EAN-13'),
                    
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('tipoContenido',$this->tipoContenido);

		$criteria->compare('fabricante',$this->fabricante);

		$criteria->compare('cerveceria',$this->cerveceria);

		$criteria->compare('numPedido',$this->numPedido,true);

		$criteria->compare('numComision',$this->numComision,true);

		$criteria->compare('ubicacionT',$this->ubicacionT);

		$criteria->compare('descripcion',$this->descripcion,true);

		$criteria->compare('titulo',$this->titulo,true);

		$criteria->compare('version',$this->version,true);

		$criteria->compare('medio',$this->medio);

		$criteria->compare('idioma',$this->idioma);

		$criteria->compare('disponibles',$this->disponibles);

		$criteria->compare('existencias',$this->existencias);

		$criteria->compare('modulo',$this->modulo);

		$criteria->compare('columna',$this->columna);

		$criteria->compare('fila',$this->fila);

		$criteria->compare('documento',$this->documento);

		$criteria->compare('ruta',$this->ruta,true);

		$criteria->compare('fechaCreacion',$this->fechaCreacion,true);

		$criteria->compare('fechaRecepcion',$this->fechaRecepcion,true);

		$criteria->compare('autores',$this->autores,true);

		$criteria->compare('usuario',$this->usuario);

		$criteria->compare('revisado',$this->revisado);

		$criteria->compare('userRevisado',$this->userRevisado);

		$criteria->compare('fechaRevisado',$this->fechaRevisado,true);

		$criteria->compare('eliminado',$this->eliminado);

		$criteria->compare('secuencia',$this->secuencia);

		$criteria->compare('ordenSecuencia',$this->ordenSecuencia);
                
                $criteria->compare('ISBN',$this->ISBN,true);

		$criteria->compare('EAN13',$this->EAN13,true);
                
                $criteria->compare('UT_Area',$this->UT_Area,true);
                
                $criteria->compare('UT_Proceso',$this->UT_Proceso,true);
                
                $criteria->compare('UT_Equipo',$this->UT_Equipo,true);
                
                $criteria->compare('UT_Tablero_TAG',$this->UT_Tablero_TAG,true);
                
                $criteria->compare('UT_Motor_TAG',$this->UT_Motor_TAG,true);
                

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
